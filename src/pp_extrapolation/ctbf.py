"""Contracted time-to-boundary flow for scalar ordered degradation.

The network predicts a strictly positive local degradation velocity. Remaining
time is not an unconstrained regression output: it is the numerical integral
of inverse velocity from a fixed failure boundary to the observed health.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class VelocityField(nn.Module):
    def __init__(self, dimension: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CTBFNet(nn.Module):
    def __init__(
        self,
        dimension: int,
        width: int,
        boundary: float,
        quadrature_points: int,
        *,
        weak_rate_prior: bool,
        residual_bound: float,
        rate_floor: float,
        center: np.ndarray,
        scale: np.ndarray,
        rate_indices: tuple[int, ...] = (1,),
        rate_aggregation: str = "first",
    ):
        super().__init__()
        if quadrature_points < 2:
            raise ValueError("quadrature_points must be at least two")
        self.velocity_field = VelocityField(dimension, width)
        self.boundary = float(boundary)
        self.quadrature_points = int(quadrature_points)
        self.weak_rate_prior = bool(weak_rate_prior)
        self.residual_bound = float(residual_bound)
        self.rate_floor = float(rate_floor)
        if weak_rate_prior and not rate_indices:
            raise ValueError("weak-rate prior requires a rate index")
        if rate_aggregation not in {"first", "median"}:
            raise ValueError("rate_aggregation must be first or median")
        self.rate_indices = tuple(int(value) for value in rate_indices)
        if self.rate_indices and (
            min(self.rate_indices) < 0 or max(self.rate_indices) >= dimension
        ):
            raise ValueError("rate index is outside the feature dimension")
        self.rate_aggregation = rate_aggregation
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer(
            "quadrature_grid",
            torch.linspace(0.0, 1.0, self.quadrature_points),
        )

    def _log_velocity(
        self, raw_x: torch.Tensor, standardized_x: torch.Tensor
    ) -> torch.Tensor:
        correction = self.velocity_field(standardized_x)
        if not self.weak_rate_prior:
            return correction
        rates = torch.stack(
            [-raw_x[..., index] for index in self.rate_indices], dim=-1
        )
        rates = torch.clamp(rates, min=self.rate_floor)
        if self.rate_aggregation == "median":
            base = torch.median(rates, dim=-1).values
        else:
            base = rates[..., 0]
        return torch.log(base) + self.residual_bound * torch.tanh(correction)

    def local_velocity(self, raw_x: torch.Tensor) -> torch.Tensor:
        z = (raw_x - self.center) / self.scale
        log_velocity = torch.clamp(self._log_velocity(raw_x, z), -12.0, 4.0)
        return torch.exp(log_velocity)

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        health = raw_x[:, 0]
        span = torch.clamp(health - self.boundary, min=0.0)
        q = self.quadrature_grid.view(1, -1)
        integrated_health = self.boundary + span[:, None] * q
        path = raw_x[:, None, :].repeat(1, self.quadrature_points, 1)
        path[..., 0] = integrated_health
        z = (path - self.center) / self.scale
        log_velocity = torch.clamp(self._log_velocity(path, z), -12.0, 4.0)
        inverse_velocity = torch.exp(-log_velocity)
        integral = torch.trapezoid(inverse_velocity, q, dim=1) * span
        return torch.clamp(integral, min=0.0)


@dataclass
class CTBFFit:
    model: CTBFNet
    selection: dict


def _arrays(split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(split["x"], dtype=np.float32)
    y = np.asarray(split["y"], dtype=np.float32)
    groups = np.asarray(split["groups"])
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != (len(x),):
        raise ValueError("unaligned CTBF split")
    if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid CTBF split")
    return x, y, groups


def fit_ctbf(
    train: dict,
    validation: dict,
    *,
    seed: int = 42,
    width: int = 32,
    boundary: float = 0.8,
    quadrature_points: int = 24,
    weak_rate_prior: bool = True,
    residual_bound: float = 1.0,
    rate_indices: tuple[int, ...] = (1,),
    rate_aggregation: str = "first",
    velocity_supervision_weight: float = 0.05,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    max_epochs: int = 350,
    patience: int = 60,
    batch_size: int = 512,
) -> CTBFFit:
    tx, ty, tg = _arrays(train)
    vx, vy, _ = _arrays(validation)
    if np.any(tx[:, 0] <= boundary):
        raise ValueError("training health must exceed the failure boundary")
    center = tx.mean(axis=0)
    scale = tx.std(axis=0)
    scale[scale < 1e-8] = 1.0
    if weak_rate_prior and not rate_indices:
        raise ValueError("weak-rate prior requires rate_indices")
    if velocity_supervision_weight > 0 and not rate_indices:
        raise ValueError("velocity supervision requires rate_indices")
    observed_rates = (
        -tx[:, np.asarray(rate_indices)]
        if rate_indices
        else np.empty((len(tx), 0), dtype=np.float32)
    )
    positive_rates = observed_rates[observed_rates > 1e-8]
    rate_floor = (
        float(np.quantile(positive_rates, 0.10))
        if len(positive_rates)
        else 1e-4
    )

    torch.manual_seed(int(seed))
    model = CTBFNet(
        tx.shape[1],
        int(width),
        boundary,
        quadrature_points,
        weak_rate_prior=weak_rate_prior,
        residual_bound=residual_bound,
        rate_floor=max(rate_floor, 1e-6),
        rate_indices=rate_indices,
        rate_aggregation=rate_aggregation,
        center=center,
        scale=scale,
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
    target_scale = max(float(np.std(ty)), 1.0)
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tx))
        for start in range(0, len(tx), batch_size):
            index = torch.as_tensor(order[start : start + batch_size])
            prediction = model(x_train[index])
            rul_loss = ((prediction - y_train[index]) / target_scale) ** 2
            loss = torch.mean(weights[index] * rul_loss)

            # A weak local constraint anchors velocity where a negative
            # transition was observed; RUL supervision remains primary.
            if velocity_supervision_weight > 0:
                rates = torch.stack(
                    [-x_train[index, value] for value in model.rate_indices], dim=1
                )
                rates = torch.clamp(rates, min=model.rate_floor)
                if model.rate_aggregation == "median":
                    rate = torch.median(rates, dim=1).values
                else:
                    rate = rates[:, 0]
                valid = rate > model.rate_floor
                if torch.any(valid):
                    predicted_rate = model.local_velocity(x_train[index][valid])
                    rate_loss = torch.nn.functional.smooth_l1_loss(
                        torch.log(predicted_rate),
                        torch.log(rate[valid]),
                    )
                    loss = loss + velocity_supervision_weight * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation).cpu().numpy()
        score = float(np.sqrt(np.mean((validation_prediction - vy) ** 2)))
        history.append({"epoch": epoch, "validation_rmse": score})
        if score < best - 1e-8:
            best = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break

    model.load_state_dict(best_state)
    return CTBFFit(
        model=model,
        selection={
            "seed": int(seed),
            "width": int(width),
            "boundary": float(boundary),
            "quadrature_points": int(quadrature_points),
            "weak_rate_prior": bool(weak_rate_prior),
            "residual_bound": float(residual_bound),
            "rate_indices": list(rate_indices),
            "rate_aggregation": rate_aggregation,
            "velocity_supervision_weight": float(velocity_supervision_weight),
            "rate_floor": float(model.rate_floor),
            "selected_epoch": int(best_epoch),
            "validation_rmse": float(best),
            "history": history,
        },
    )


def predict_ctbf(fit: CTBFFit, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    fit.model.eval()
    with torch.no_grad():
        prediction = fit.model(torch.as_tensor(values)).cpu().numpy()
    return np.asarray(prediction, dtype=np.float64)


def predict_rate_quotient(
    x: np.ndarray, *, boundary: float = 0.8, rate_floor: float = 1e-4
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    span = np.maximum(values[:, 0] - boundary, 0.0)
    rate = np.maximum(-values[:, 1], rate_floor)
    return span / rate


def contract_normalize_features(
    x: np.ndarray,
    boundaries: np.ndarray,
    *,
    rate_indices: tuple[int, ...],
    mean_indices: tuple[int, ...] = (),
    std_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Map unit-specific failure boundaries to zero without label access."""
    values = np.asarray(x, dtype=np.float64)
    boundary = np.asarray(boundaries, dtype=np.float64)
    if values.ndim != 2 or boundary.shape != (len(values),):
        raise ValueError("unaligned contract-normalization arrays")
    if not np.isfinite(values).all() or not np.isfinite(boundary).all():
        raise ValueError("contract normalization requires finite values")
    denominator = 1.0 - boundary
    if np.any(denominator <= 1e-8):
        raise ValueError("failure boundary must be below initial health")
    transformed = values.copy()
    transformed[:, 0] = (values[:, 0] - boundary) / denominator
    for index in rate_indices:
        transformed[:, index] = values[:, index] / denominator
    for index in mean_indices:
        transformed[:, index] = (values[:, index] - boundary) / denominator
    for index in std_indices:
        transformed[:, index] = values[:, index] / denominator
    return transformed.astype(np.float32)
