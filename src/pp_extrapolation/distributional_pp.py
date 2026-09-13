"""Distributional structural prior-residual network for PP-X."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import (
    equal_group_weights,
    select_affine_initialization,
    transform_features,
)


class DistributionalPPNet(nn.Module):
    """Frozen affine prior plus a bounded stochastic residual distribution."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        residual_bound: float,
        initial_trust: float = 0.5,
    ) -> None:
        super().__init__()
        self.residual_bound = float(residual_bound)
        self.affine = nn.Linear(input_dim, 1)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.location = nn.Linear(width, 1)
        self.log_scale = nn.Linear(width, 1)
        self.trust = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.location.weight)
        nn.init.zeros_(self.location.bias)
        nn.init.constant_(self.log_scale.bias, -2.0)
        nn.init.zeros_(self.trust.weight)
        nn.init.constant_(
            self.trust.bias,
            float(np.log(initial_trust / (1.0 - initial_trust))),
        )

    def parameters_to_optimize(self):
        return (
            list(self.trunk.parameters())
            + list(self.location.parameters())
            + list(self.log_scale.parameters())
            + list(self.trust.parameters())
        )

    def distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        authority = 1.0 - torch.sigmoid(self.trust(x)).squeeze(1)
        location = self.affine(x).squeeze(1) + authority * self.residual_bound * torch.tanh(
            self.location(hidden).squeeze(1)
        )
        scale = authority * 0.25 * self.residual_bound * torch.nn.functional.softplus(
            self.log_scale(hidden).squeeze(1)
        )
        return location, scale

    def samples(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        location, scale = self.distribution(x)
        return location[:, None] + scale[:, None] * noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution(x)[0]


@dataclass
class DistributionalPPFit:
    model: DistributionalPPNet
    center: np.ndarray
    scale: np.ndarray
    target_scale: float
    selected_epoch: int
    validation_mse: float
    config: dict


def _arrays(split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(split["x"], dtype=np.float64)
    y = np.asarray(split["y"], dtype=np.float64)
    groups = np.asarray(split["groups"])
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape:
        raise ValueError("split arrays must align")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("split arrays must be finite and nonempty")
    return x, y, groups


def fit_distributional_pp(
    train: dict,
    validation: dict,
    *,
    seed: int,
    width: int = 32,
    residual_bound: float = 0.5,
    mean_weight: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    max_epochs: int = 300,
    patience: int = 50,
    affine_selection: dict | None = None,
    restore_best: bool = True,
) -> DistributionalPPFit:
    """Fit with a group-balanced energy score and validation-only stopping."""
    train_x, train_y, groups = _arrays(train)
    validation_x, validation_y, _ = _arrays(validation)
    selection = affine_selection or select_affine_initialization(train, validation)
    center = np.asarray(selection["center"])
    scale = np.asarray(selection["scale"])
    target_scale = float(selection["target_scale"])
    initialization = selection["initialization"]
    torch.manual_seed(int(seed))
    model = DistributionalPPNet(
        train_x.shape[1], int(width), float(residual_bound),
    )
    with torch.no_grad():
        model.affine.weight.copy_(torch.as_tensor(initialization.weight)[None, :])
        model.affine.bias.copy_(torch.as_tensor([initialization.bias]))
    model.affine.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters_to_optimize(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    x = torch.as_tensor(transform_features(train_x, center, scale), dtype=torch.float32)
    y = torch.as_tensor(train_y / target_scale, dtype=torch.float32)
    weights = torch.as_tensor(equal_group_weights(groups), dtype=torch.float32)
    vx = torch.as_tensor(
        transform_features(validation_x, center, scale), dtype=torch.float32
    )
    vy = torch.as_tensor(validation_y / target_scale, dtype=torch.float32)
    rng = np.random.default_rng(int(seed))
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), 512):
            index = torch.as_tensor(order[start:start + 512])
            noise = torch.randn(len(index), 8)
            samples = model.samples(x[index], noise)
            first = torch.mean(torch.abs(samples - y[index, None]), dim=1)
            pairwise = torch.mean(
                torch.abs(samples[:, :, None] - samples[:, None, :]), dim=(1, 2)
            )
            mean_error = (model(x[index]) - y[index]).square()
            loss = torch.mean(
                weights[index]
                * (first - 0.5 * pairwise + float(mean_weight) * mean_error)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters_to_optimize(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = torch.clamp(model(vx), 0.0, 1.0)
            current = float(torch.mean((validation_prediction - vy).square()))
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
            best = float(torch.mean((torch.clamp(model(vx), 0.0, 1.0) - vy).square()))
    return DistributionalPPFit(
        model=model,
        center=center,
        scale=scale,
        target_scale=target_scale,
        selected_epoch=max(int(best_epoch), 1),
        validation_mse=best * target_scale**2,
        config={
            "width": int(width),
            "residual_bound": float(residual_bound),
            "mean_weight": float(mean_weight),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        },
    )


def predict_distributional_pp(fit: DistributionalPPFit, x: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(
        transform_features(x, fit.center, fit.scale), dtype=torch.float32
    )
    fit.model.eval()
    with torch.no_grad():
        prediction = fit.model(value).cpu().numpy() * fit.target_scale
    return np.clip(prediction, 0.0, fit.target_scale)


def sample_distributional_pp(
    fit: DistributionalPPFit,
    x: np.ndarray,
    *,
    samples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    value = torch.as_tensor(
        transform_features(x, fit.center, fit.scale), dtype=torch.float32
    )
    noise = torch.as_tensor(
        np.random.default_rng(seed).standard_normal((len(value), int(samples))),
        dtype=torch.float32,
    )
    fit.model.eval()
    with torch.no_grad():
        draws = fit.model.samples(value, noise).cpu().numpy() * fit.target_scale
    return np.clip(draws, 0.0, fit.target_scale)
