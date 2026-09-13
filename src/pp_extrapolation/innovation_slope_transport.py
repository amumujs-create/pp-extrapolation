"""Censoring-aligned stochastic innovation slope transport for PP-X."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from scipy.special import ndtri

from .model import equal_group_weights


class InnovationSlopeTransportNet(nn.Module):
    """Boundary anchor plus stochastic positive-rate integral and direct route."""

    def __init__(self, input_dim: int, progress_index: int, width: int, steps: int,
                 route: str = "hybrid"):
        super().__init__()
        if route not in ("hybrid", "direct_only", "flow_only"):
            raise ValueError("unknown CIST route")
        self.route = route
        self.progress_index = int(progress_index)
        self.steps = int(steps)
        context_dim = input_dim - 1
        self.context = nn.Sequential(
            nn.Linear(context_dim, width), nn.ReLU(),
            nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
        )
        self.anchor = nn.Linear(width, 1)
        self.rate = nn.Sequential(
            nn.Linear(width + 3, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1),
        )
        self.direct = nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1),
        )
        self.gate = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if route == "direct_only":
            for module in (self.context, self.anchor, self.rate, self.gate):
                module.requires_grad_(False)
        elif route == "flow_only":
            self.direct.requires_grad_(False)
            self.gate.requires_grad_(False)

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (x[:, :self.progress_index], x[:, self.progress_index + 1:]), dim=1
        )

    def samples(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 3 or noise.shape[0] != len(x) or noise.shape[2] != self.steps:
            raise ValueError("noise must have shape (rows, samples, integration_steps)")
        if self.route == "direct_only":
            return self.direct(x).expand(-1, noise.shape[1])
        context = self.context(self._context(x))
        anchor = self.anchor(context).squeeze(1)
        progress = x[:, self.progress_index]
        sample_count = noise.shape[1]
        midpoint = (
            torch.arange(self.steps, device=x.device, dtype=x.dtype) + 0.5
        ) / self.steps
        time = (
            progress[:, None, None] * midpoint[None, None, :]
        ).expand(-1, sample_count, -1)
        hidden = context[:, None, None, :].expand(
            -1, sample_count, self.steps, -1
        )
        rate_input = torch.cat(
            (hidden, time[..., None], noise[..., None], (time * noise)[..., None]),
            dim=3,
        )
        rate = torch.nn.functional.softplus(self.rate(rate_input).squeeze(3))
        flow = anchor[:, None] - progress[:, None] * torch.mean(rate, dim=2)
        if self.route == "flow_only":
            return flow
        direct = self.direct(x).squeeze(1)
        inside = ((progress >= 0.0) & (progress <= 1.0)).to(x.dtype)
        outside_distance = torch.relu(-progress) + torch.relu(progress - 1.0)
        direct_authority = torch.exp(-2.0 * outside_distance)
        flow_gate = torch.sigmoid(self.gate(x)).squeeze(1)
        effective_gate = flow_gate + (1.0 - flow_gate) * (1.0 - direct_authority)
        return (
            effective_gate[:, None] * flow
            + (1.0 - effective_gate[:, None]) * direct[:, None]
            + 0.0 * inside[:, None]
        )

    def mean(self, x: torch.Tensor, samples: int = 32) -> torch.Tensor:
        count = int(samples)
        nodes = torch.as_tensor(
            ndtri((np.arange(count, dtype=np.float64) + 0.5) / count),
            dtype=x.dtype,
            device=x.device,
        )
        noise = torch.stack(
            [torch.roll(nodes, shifts=step) for step in range(self.steps)], dim=1
        )[None, :, :].expand(len(x), -1, -1)
        return self.samples(x, noise).mean(dim=1)


@dataclass
class InnovationSlopeTransportFit:
    model: InnovationSlopeTransportNet
    x_center: np.ndarray
    x_scale: np.ndarray
    progress_min: float
    progress_scale: float
    y_center: float
    y_scale: float
    progress_index: int
    selected_epoch: int
    validation_mse: float
    config: dict


def _arrays(rows: dict):
    x = np.asarray(rows["x"], dtype=np.float64)
    y = np.asarray(rows["y"], dtype=np.float64)
    groups = np.asarray(rows["groups"])
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape:
        raise ValueError("rows must align")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("rows must be finite and nonempty")
    return x, y, groups


def _fit_scale(x: np.ndarray, progress_index: int, weights: np.ndarray):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    center = np.sum(weights[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (x - center) ** 2, axis=0))
    scale[scale < 1e-8] = 1.0
    progress = x[:, progress_index]
    progress_min = float(np.min(progress))
    progress_scale = max(float(np.max(progress) - progress_min), 1e-8)
    center[progress_index] = progress_min
    scale[progress_index] = progress_scale
    return center, scale, progress_min, progress_scale


def fit_innovation_slope_transport(
    train: dict,
    validation: dict,
    *,
    progress_index: int,
    seed: int,
    width: int = 32,
    steps: int = 4,
    beta: float = 1.0,
    mean_weight: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    max_epochs: int = 300,
    patience: int = 50,
    restore_best: bool = True,
    sample_weight: np.ndarray | None = None,
    validation_weight: np.ndarray | None = None,
    mean_objective: str = "zero_noise",
    mean_samples: int = 32,
    validation_clip_to_train_range: bool = False,
    validation_samples: int = 32,
    route: str = "hybrid",
) -> InnovationSlopeTransportFit:
    if mean_objective not in ("zero_noise", "predictive_mean") or mean_samples < 1:
        raise ValueError("invalid mean objective or quadrature size")
    train_x, train_y, groups = _arrays(train)
    validation_x, validation_y, _ = _arrays(validation)
    index = int(progress_index) % train_x.shape[1]
    importance = (
        np.ones(len(train_y), dtype=np.float64)
        if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    if importance.shape != train_y.shape or np.any(importance <= 0) or not np.isfinite(importance).all():
        raise ValueError("sample_weight must be aligned, finite, and positive")
    normalized_importance = importance / np.sum(importance)
    x_center, x_scale, progress_min, progress_scale = _fit_scale(
        train_x, index, importance
    )
    y_center = float(np.sum(normalized_importance * train_y))
    y_scale = max(float(np.sqrt(np.sum(
        normalized_importance * (train_y - y_center) ** 2
    ))), 1e-8)
    x = torch.as_tensor((train_x - x_center) / x_scale, dtype=torch.float32)
    y = torch.as_tensor((train_y - y_center) / y_scale, dtype=torch.float32)
    vx = torch.as_tensor((validation_x - x_center) / x_scale, dtype=torch.float32)
    vy = torch.as_tensor((validation_y - y_center) / y_scale, dtype=torch.float32)
    combined = equal_group_weights(groups) * importance
    combined /= np.mean(combined)
    weights = torch.as_tensor(combined, dtype=torch.float32)
    validation_importance = (
        np.ones(len(validation_y), dtype=np.float64)
        if validation_weight is None
        else np.asarray(validation_weight, dtype=np.float64)
    )
    if validation_importance.shape != validation_y.shape or np.any(validation_importance <= 0) or not np.isfinite(validation_importance).all():
        raise ValueError("validation_weight must be aligned, finite, and positive")
    validation_importance /= np.mean(validation_importance)
    validation_weights = torch.as_tensor(validation_importance, dtype=torch.float32)
    torch.manual_seed(int(seed))
    model = InnovationSlopeTransportNet(train_x.shape[1], index, int(width), int(steps), route=route)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed))
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epsilon = 1e-5 if not float(beta).is_integer() else 0.0
    def validation_loss():
        prediction = model.mean(vx, samples=validation_samples)
        if validation_clip_to_train_range:
            cap = max(float(np.max(train_y)), 1.0)
            prediction = prediction.clamp(-y_center / y_scale, (cap-y_center) / y_scale)
        return float(torch.mean(validation_weights * (prediction - vy).square()))
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), 512):
            take = torch.as_tensor(order[start:start + 512])
            noise_a = torch.randn(len(take), 1, int(steps))
            noise_b = torch.randn(len(take), 1, int(steps))
            sample_a = model.samples(x[take], noise_a).squeeze(1)
            sample_b = model.samples(x[take], noise_b).squeeze(1)
            data_term = 0.5 * (
                (torch.abs(sample_a - y[take]) + epsilon).pow(beta)
                + (torch.abs(sample_b - y[take]) + epsilon).pow(beta)
            )
            diversity = (torch.abs(sample_a - sample_b) + epsilon).pow(beta)
            if mean_objective == "predictive_mean":
                point = model.mean(x[take], samples=mean_samples)
            else:
                zero_noise = torch.zeros(len(take), 8, int(steps))
                point = model.samples(x[take], zero_noise).mean(dim=1)
            mean_loss = (point - y[take]).square()
            loss = torch.mean(
                weights[take]
                * (data_term - 0.5 * diversity + float(mean_weight) * mean_loss)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            current = validation_loss()
        if current < best - 1e-10:
            best = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= int(patience):
            break
    if restore_best:
        model.load_state_dict(best_state)
    else:
        best_epoch = int(max_epochs)
        with torch.no_grad():
            best = validation_loss()
    return InnovationSlopeTransportFit(
        model, x_center, x_scale, progress_min, progress_scale,
        y_center, y_scale, index, max(int(best_epoch), 1),
        best * y_scale**2,
        {"width": int(width), "steps": int(steps), "beta": float(beta),
         "mean_weight": float(mean_weight), "learning_rate": float(learning_rate),
         "weight_decay": float(weight_decay), "mean_objective": mean_objective,
         "mean_samples": int(mean_samples),
         "validation_clip_to_train_range": bool(validation_clip_to_train_range),
         "validation_samples": int(validation_samples), "route": route},
    )


def predict_innovation_slope_transport(
    fit: InnovationSlopeTransportFit,
    x: np.ndarray,
    *,
    samples: int = 128,
    seed: int = 0,
) -> np.ndarray:
    value = torch.as_tensor(
        (np.asarray(x, dtype=np.float64) - fit.x_center) / fit.x_scale,
        dtype=torch.float32,
    )
    count = int(samples)
    nodes = ndtri((np.arange(count, dtype=np.float64) + 0.5) / count)
    noise = torch.as_tensor(
        np.stack([np.roll(nodes, step) for step in range(fit.model.steps)], axis=1),
        dtype=torch.float32,
    )[None, :, :].expand(len(value), -1, -1)
    fit.model.eval()
    with torch.no_grad():
        prediction = fit.model.samples(value, noise).mean(dim=1).numpy()
    return np.clip(prediction * fit.y_scale + fit.y_center, 0.0, None)
