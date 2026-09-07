"""Boundary-factorized PP for health-to-RUL relationship shifts.

The model learns the positive RUL-per-health-margin quotient.  Its affine
path is fitted first and frozen; a bounded neural residual learns deviations
caused by rate acceleration and other causal history features.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _softplus_inverse(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), 1e-5)
    return value + np.log(-np.expm1(-value))


def equal_dataset_unit_weights(dataset: np.ndarray, units: np.ndarray) -> np.ndarray:
    dataset, units = np.asarray(dataset), np.asarray(units)
    if dataset.shape != units.shape:
        raise ValueError("dataset and units must have the same shape")
    weights = np.zeros(len(dataset), dtype=np.float64)
    labels = np.unique(dataset)
    for label in labels:
        mask = dataset == label
        _, inverse, counts = np.unique(units[mask], return_inverse=True, return_counts=True)
        weights[np.flatnonzero(mask)] = len(dataset) / (len(labels) * len(counts) * counts[inverse])
    return weights.astype(np.float32)


class BoundaryQuotientPPNet(nn.Module):
    """Exact-zero boundary gate with a frozen affine quotient tail."""

    def __init__(self, input_dim: int, width: int = 64, residual_bound: float = 2.0):
        super().__init__()
        self.residual_bound = float(residual_bound)
        self.affine = nn.Linear(input_dim, 1)
        self.nonlinear = nn.Sequential(
            nn.Linear(input_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1),
        )
        nn.init.zeros_(self.nonlinear[-1].weight)
        nn.init.zeros_(self.nonlinear[-1].bias)

    def forward(self, value: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        correction = self.residual_bound * torch.tanh(self.nonlinear(value))
        quotient = torch.nn.functional.softplus(self.affine(value) + correction)
        return torch.clamp(margin, min=0.0) * quotient.squeeze(1)


@dataclass
class BoundaryQuotientFit:
    model: BoundaryQuotientPPNet
    center: np.ndarray
    scale: np.ndarray
    selection: dict


def _validate(rows: dict) -> None:
    n = len(rows["y"])
    if np.asarray(rows["x"]).ndim != 2 or np.asarray(rows["x"]).shape[0] != n:
        raise ValueError("x must have shape (n,d)")
    for key in ("margin", "y", "units", "dataset"):
        if np.asarray(rows[key]).shape != (n,):
            raise ValueError(f"{key} must have shape (n,)")
    if np.any(np.asarray(rows["margin"]) < 0) or not np.isfinite(np.asarray(rows["x"])).all():
        raise ValueError("features must be finite and margin nonnegative")


def _affine_initialization(rows: dict, center: np.ndarray, scale: np.ndarray, alpha: float):
    x = (np.asarray(rows["x"], dtype=np.float64) - center) / scale
    margin = np.maximum(np.asarray(rows["margin"], dtype=np.float64), 1e-4)
    latent = _softplus_inverse(np.asarray(rows["y"], dtype=np.float64) / margin)
    weight = equal_dataset_unit_weights(rows["dataset"], rows["units"]).astype(np.float64)
    total = weight.sum()
    xc = np.sum(weight[:, None] * x, axis=0) / total
    yc = np.sum(weight * latent) / total
    xx, yy = x - xc, latent - yc
    coefficient = np.linalg.solve(xx.T @ (weight[:, None] * xx) + alpha * np.eye(x.shape[1]), xx.T @ (weight * yy))
    return coefficient.astype(np.float32), float(yc - xc @ coefficient)


def fit_boundary_quotient_pp(
    train: dict, validation: dict, *, seed: int, width: int = 64,
    alpha: float = 10.0, learning_rate: float = 1e-3,
    weight_decay: float = 1e-2, max_epochs: int = 500, patience: int = 70,
    batch_size: int = 512, residual_bound: float = 2.0, restore_best: bool = True,
    trainable_affine: bool = False,
) -> BoundaryQuotientFit:
    _validate(train); _validate(validation)
    center = np.asarray(train["x"], dtype=np.float64).mean(0)
    scale = np.asarray(train["x"], dtype=np.float64).std(0)
    scale[scale < 1e-6] = 1.0
    coefficient, bias = _affine_initialization(train, center, scale, float(alpha))
    torch.manual_seed(int(seed))
    model = BoundaryQuotientPPNet(np.asarray(train["x"]).shape[1], int(width), residual_bound)
    with torch.no_grad():
        model.affine.weight.copy_(torch.tensor(coefficient)[None, :])
        model.affine.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    model.affine.requires_grad_(bool(trainable_affine))
    parameters = list(model.nonlinear.parameters())
    if trainable_affine:
        parameters += list(model.affine.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    x = torch.tensor((np.asarray(train["x"]) - center) / scale, dtype=torch.float32)
    margin = torch.tensor(train["margin"], dtype=torch.float32)
    y = torch.tensor(train["y"], dtype=torch.float32)
    weights = torch.tensor(equal_dataset_unit_weights(train["dataset"], train["units"]))
    vx = torch.tensor((np.asarray(validation["x"]) - center) / scale, dtype=torch.float32)
    vm = torch.tensor(validation["margin"], dtype=torch.float32)
    vy = np.asarray(validation["y"], dtype=np.float64)
    vd = np.asarray(validation["dataset"])
    rng = np.random.default_rng(seed)

    def validation_loss():
        model.eval()
        with torch.no_grad(): prediction = model(vx, vm).numpy()
        return float(np.mean([np.mean((prediction[vd == d] - vy[vd == d]) ** 2) for d in np.unique(vd)]))

    best, best_epoch, state = validation_loss(), 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            index = torch.tensor(order[start:start + batch_size])
            prediction = model(x[index], margin[index])
            loss = torch.mean(weights[index] * (prediction - y[index]).square())
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        current = validation_loss()
        if current < best - 1e-8:
            best, best_epoch, state = current, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break
    if restore_best:
        model.load_state_dict(state)
    else:
        best_epoch = int(max_epochs)
    return BoundaryQuotientFit(model, center.astype(np.float32), scale.astype(np.float32), {
        "seed": int(seed), "width": int(width), "alpha": float(alpha),
        "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
        "selected_epoch": int(best_epoch), "validation_dataset_macro_mse": float(best),
        "residual_bound": float(residual_bound), "restore_best": bool(restore_best),
        "trainable_affine": bool(trainable_affine),
    })


def predict_boundary_quotient(fit: BoundaryQuotientFit, rows: dict) -> np.ndarray:
    x = torch.tensor((np.asarray(rows["x"]) - fit.center) / fit.scale, dtype=torch.float32)
    margin = torch.tensor(rows["margin"], dtype=torch.float32)
    fit.model.eval()
    with torch.no_grad(): return fit.model(x, margin).numpy().astype(np.float64)


def predict_boundary_affine(fit: BoundaryQuotientFit, rows: dict) -> np.ndarray:
    x = torch.tensor((np.asarray(rows["x"]) - fit.center) / fit.scale, dtype=torch.float32)
    margin = torch.tensor(rows["margin"], dtype=torch.float32)
    with torch.no_grad():
        quotient = torch.nn.functional.softplus(fit.model.affine(x)).squeeze(1)
        return (margin * quotient).numpy().astype(np.float64)
