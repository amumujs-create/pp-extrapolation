"""Differentiable weak-prior projection with learned residual authority."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class WeakPriorProjection(nn.Module):
    """Enforce nonnegative monotone rays and blend with exact raw fallback."""

    def forward(
        self,
        raw: torch.Tensor,
        authority: torch.Tensor,
        minimum_increment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.ndim != 2:
            raise ValueError("raw predictions must be [batch, ordered_point]")
        if authority.ndim == 1:
            authority = authority[:, None]
        if authority.shape != (len(raw), 1):
            raise ValueError("authority must have one value per ray")
        if torch.any((authority < 0) | (authority > 1)):
            raise ValueError("authority must lie in [0, 1]")
        if minimum_increment is None:
            minimum_increment = torch.zeros(
                (len(raw), 1), dtype=raw.dtype, device=raw.device
            )
        elif minimum_increment.ndim == 1:
            minimum_increment = minimum_increment[:, None]
        if (
            minimum_increment.shape != (len(raw), 1)
            or torch.any(minimum_increment < 0)
        ):
            raise ValueError("minimum increment must be nonnegative per ray")
        step = torch.arange(
            raw.shape[1], dtype=raw.dtype, device=raw.device
        )[None, :]
        ramp = minimum_increment * step
        adjusted = torch.maximum(raw - ramp, -ramp)
        feasible = torch.cummax(adjusted, dim=1).values + ramp
        prediction = raw + authority * (feasible - raw)
        return prediction, feasible


class WeakPriorAuthorityHead(nn.Module):
    """Map source-only falsification evidence to continuous authority."""

    def __init__(self, dimension: int, width: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


@dataclass
class WeakPriorAuthorityFit:
    model: WeakPriorAuthorityHead
    center: np.ndarray
    scale: np.ndarray
    best_epoch: int
    validation_loss: float


def fit_authority_head(
    features: np.ndarray,
    oracle_authority: np.ndarray,
    *,
    seed: int = 20260912,
    width: int = 24,
    max_epochs: int = 2000,
    patience: int = 150,
) -> WeakPriorAuthorityFit:
    """Fit with a fixed task split; confirmation tasks never enter."""
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(oracle_authority, dtype=np.float64)
    if (
        features.ndim != 2
        or target.shape != (len(features),)
        or len(features) < 20
        or np.any((target < 0) | (target > 1))
    ):
        raise ValueError("invalid authority training data")
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = torch.as_tensor((features - center) / scale, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    split = max(1, int(0.80 * len(order)))
    train_index, validation_index = order[:split], order[split:]
    torch.manual_seed(seed)
    model = WeakPriorAuthorityHead(features.shape[1], width)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=0.05
    )
    best = float("inf")
    best_epoch = 0
    state = copy.deepcopy(model.state_dict())
    for epoch in range(1, max_epochs + 1):
        model.train()
        estimate = model(x[train_index])
        loss = torch.mean((estimate - y[train_index]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(torch.mean(
                (model(x[validation_index]) - y[validation_index]) ** 2
            ))
        if value < best - 1e-8:
            best = value
            best_epoch = epoch
            state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return WeakPriorAuthorityFit(
        model=model,
        center=center,
        scale=scale,
        best_epoch=best_epoch,
        validation_loss=best,
    )


def predict_authority(
    fit: WeakPriorAuthorityFit, features: np.ndarray
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(fit.center):
        raise ValueError("features do not match fitted authority head")
    x = torch.as_tensor(
        (features - fit.center) / fit.scale, dtype=torch.float32
    )
    with torch.no_grad():
        return fit.model(x).cpu().numpy().astype(np.float64)


def apply_weak_prior_projection(
    raw: np.ndarray,
    authority: np.ndarray,
    minimum_increment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-facing inference wrapper for ordered prediction rays."""
    raw_tensor = torch.as_tensor(raw, dtype=torch.float32)
    authority_tensor = torch.as_tensor(authority, dtype=torch.float32)
    increment_tensor = (
        None if minimum_increment is None
        else torch.as_tensor(minimum_increment, dtype=torch.float32)
    )
    with torch.no_grad():
        prediction, feasible = WeakPriorProjection()(
            raw_tensor, authority_tensor, increment_tensor
        )
    return (
        prediction.cpu().numpy().astype(np.float64),
        feasible.cpu().numpy().astype(np.float64),
    )
