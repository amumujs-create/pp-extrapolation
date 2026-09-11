"""Contract-conditioned observed or latent event-coordinate flow."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class EventCoordinateFlow(nn.Module):
    def __init__(
        self,
        dimension: int,
        width: int,
        *,
        observed_coordinate: bool,
        quadrature_points: int,
        center: np.ndarray,
        scale: np.ndarray,
        margin_scale: float,
        initial_time_scale: float,
    ):
        super().__init__()
        self.observed_coordinate = bool(observed_coordinate)
        self.margin_scale = float(margin_scale)
        self.context = nn.Sequential(
            nn.Linear(dimension, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.margin_head = nn.Linear(width, 1)
        self.velocity = nn.Sequential(
            nn.Linear(width + 1, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.margin_head.weight)
        nn.init.zeros_(self.margin_head.bias)
        nn.init.zeros_(self.velocity[-1].weight)
        initial_velocity = max(self.margin_scale / max(initial_time_scale, 1e-6), 1e-6)
        nn.init.constant_(self.velocity[-1].bias, float(np.log(initial_velocity)))
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer(
            "quadrature_grid",
            torch.linspace(0.0, 1.0, int(quadrature_points)),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.context((x - self.center) / self.scale)

    def margin(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.observed_coordinate:
            return torch.clamp(x[:, 0], min=0.0)
        if context is None:
            context = self.encode(x)
        return self.margin_scale * torch.nn.functional.softplus(
            self.margin_head(context).squeeze(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.encode(x)
        margin = self.margin(x, context)
        q = self.quadrature_grid.view(1, -1)
        coordinate = margin[:, None] * q
        normalized_coordinate = coordinate / max(self.margin_scale, 1e-8)
        repeated_context = context[:, None, :].expand(
            -1, len(self.quadrature_grid), -1
        )
        velocity_input = torch.cat(
            [normalized_coordinate[..., None], repeated_context], dim=-1
        )
        log_velocity = torch.clamp(
            self.velocity(velocity_input).squeeze(-1), -12.0, 8.0
        )
        inverse_velocity = torch.exp(-log_velocity)
        prediction = torch.trapezoid(inverse_velocity, q, dim=1) * margin
        return torch.clamp(prediction, min=0.0)


@dataclass
class EventFlowFit:
    model: EventCoordinateFlow
    selection: dict


def adjacent_group_pairs(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups)
    pairs = [
        (index, index + 1)
        for index in range(len(values) - 1)
        if values[index] == values[index + 1]
    ]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def fit_event_flow(
    train: dict,
    validation: dict,
    *,
    observed_coordinate: bool,
    seed: int = 42,
    width: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    quadrature_points: int = 24,
    order_weight: float = 0.1,
    gauge_weight: float = 0.01,
    max_epochs: int = 300,
    patience: int = 50,
    batch_size: int = 512,
) -> EventFlowFit:
    tx = np.asarray(train["x"], dtype=np.float32)
    ty = np.asarray(train["y"], dtype=np.float32)
    tg = np.asarray(train["groups"])
    vx = np.asarray(validation["x"], dtype=np.float32)
    vy = np.asarray(validation["y"], dtype=np.float32)
    if tx.ndim != 2 or len(tx) == 0 or ty.shape != (len(tx),):
        raise ValueError("invalid event-flow training split")
    if observed_coordinate and np.any(tx[:, 0] <= 0):
        raise ValueError("observed train event margins must be positive")
    center = tx.mean(axis=0)
    scale = tx.std(axis=0)
    scale[scale < 1e-8] = 1.0
    margin_scale = (
        max(float(np.median(tx[:, 0])), 1e-5)
        if observed_coordinate
        else 1.0
    )
    torch.manual_seed(int(seed))
    model = EventCoordinateFlow(
        tx.shape[1],
        int(width),
        observed_coordinate=observed_coordinate,
        quadrature_points=quadrature_points,
        center=center,
        scale=scale,
        margin_scale=margin_scale,
        initial_time_scale=max(float(np.median(ty)), 1.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x_train = torch.as_tensor(tx)
    y_train = torch.as_tensor(ty)
    x_validation = torch.as_tensor(vx)
    _, inverse, counts = np.unique(tg, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(1.0 / counts[inverse], dtype=torch.float32)
    weights /= weights.mean()
    pairs = adjacent_group_pairs(tg)
    pair_tensor = torch.as_tensor(pairs)
    target_scale = max(float(np.std(ty)), 1.0)
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tx))
        for start in range(0, len(tx), batch_size):
            index = torch.as_tensor(order[start : start + batch_size])
            prediction = model(x_train[index])
            loss = torch.mean(
                weights[index] * ((prediction - y_train[index]) / target_scale) ** 2
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        if not observed_coordinate:
            model.train()
            context = model.encode(x_train)
            margin = model.margin(x_train, context)
            structural_loss = gauge_weight * (
                margin.mean() / model.margin_scale - 1.0
            ) ** 2
            if len(pair_tensor):
                structural_loss = structural_loss + order_weight * torch.mean(
                    torch.relu(
                        margin[pair_tensor[:, 1]] - margin[pair_tensor[:, 0]]
                    )
                    / model.margin_scale
                )
            optimizer.zero_grad()
            structural_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_validation).cpu().numpy()
        score = float(np.sqrt(np.mean((val_prediction - vy) ** 2)))
        history.append({"epoch": epoch, "validation_rmse": score})
        if score < best - 1e-8:
            best = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break

    model.load_state_dict(best_state)
    return EventFlowFit(
        model,
        {
            "seed": int(seed),
            "width": int(width),
            "observed_coordinate": bool(observed_coordinate),
            "order_weight": float(order_weight),
            "gauge_weight": float(gauge_weight),
            "selected_epoch": int(best_epoch),
            "validation_rmse": float(best),
            "history": history,
        },
    )


def predict_event_flow(fit: EventFlowFit, x: np.ndarray) -> np.ndarray:
    fit.model.eval()
    with torch.no_grad():
        return (
            fit.model(torch.as_tensor(np.asarray(x, dtype=np.float32)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def latent_margin_violation_rate(fit: EventFlowFit, split: dict) -> float:
    if fit.model.observed_coordinate:
        return 0.0
    x = torch.as_tensor(np.asarray(split["x"], dtype=np.float32))
    pairs = adjacent_group_pairs(np.asarray(split["groups"]))
    if not len(pairs):
        return float("nan")
    fit.model.eval()
    with torch.no_grad():
        margin = fit.model.margin(x).cpu().numpy()
    return float(np.mean(margin[pairs[:, 1]] > margin[pairs[:, 0]]))
