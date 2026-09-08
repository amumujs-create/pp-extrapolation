"""Log-boundary-quotient PP for heterogeneous lifetime scales.

The ordinary boundary quotient learns ``RUL / margin`` on its raw scale.  That
quantity can vary by orders of magnitude across units even when every unit has
the same known failure boundary.  This module learns ``log1p(RUL / margin)``
instead and maps the prediction back through the known boundary.  The boundary
condition is therefore exact, while the regression problem is scale stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .model import PPFit, fit_pp, predict, select_affine_initialization


@dataclass
class LogBoundaryQuotientFit:
    latent_fit: PPFit
    margin_floor: float


def _latent_rows(rows: dict, margin_floor: float) -> dict:
    margin = np.asarray(rows["margin"], dtype=np.float64)
    y = np.asarray(rows["y"], dtype=np.float64)
    if margin.shape != y.shape or np.any(margin < 0) or np.any(y < 0):
        raise ValueError("margin and y must be aligned, finite, and nonnegative")
    if not np.isfinite(margin).all() or not np.isfinite(y).all():
        raise ValueError("margin and y must be finite")
    return {
        "x": np.asarray(rows["x"], dtype=np.float64),
        "y": np.log1p(y / np.maximum(margin, margin_floor)),
        "groups": np.asarray(rows["groups"]),
    }


def fit_log_boundary_quotient_pp(
    train: dict,
    validation: dict,
    *,
    seed: int,
    alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    selected_alpha: float | None = None,
    margin_floor: float = 0.05,
    max_epochs: int = 300,
    patience: int = 50,
    width: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 2.0,
) -> LogBoundaryQuotientFit:
    """Fit one affine-plus-NN PP in log quotient space.

    Alpha and the NN stopping epoch are selected without test labels.  Passing
    ``selected_alpha`` is useful for a final refit after development selection.
    """
    if not np.isfinite(margin_floor) or margin_floor <= 0:
        raise ValueError("margin_floor must be finite and positive")
    latent_train = _latent_rows(train, float(margin_floor))
    latent_validation = _latent_rows(validation, float(margin_floor))
    choices = (float(selected_alpha),) if selected_alpha is not None else tuple(alphas)
    affine = select_affine_initialization(latent_train, latent_validation, alphas=choices)
    fitted = fit_pp(
        latent_train,
        latent_validation,
        seed=seed,
        affine_selection=affine,
        max_epochs=max_epochs,
        patience=patience,
        width=width,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    fitted.selection["quotient_transform"] = "log1p_rul_over_margin"
    fitted.selection["margin_floor"] = float(margin_floor)
    return LogBoundaryQuotientFit(fitted, float(margin_floor))


def predict_log_boundary_quotient(fit: LogBoundaryQuotientFit, rows: dict) -> np.ndarray:
    margin = np.maximum(np.asarray(rows["margin"], dtype=np.float64), 0.0)
    latent = predict(fit.latent_fit, np.asarray(rows["x"], dtype=np.float64))
    return margin * np.expm1(latent)


def predict_log_boundary_affine(fit: LogBoundaryQuotientFit, rows: dict) -> np.ndarray:
    base = fit.latent_fit
    x = torch.as_tensor((np.asarray(rows["x"]) - base.center) / base.scale, dtype=torch.float32)
    base.model.eval()
    with torch.no_grad():
        latent = base.model.affine(x).squeeze(1).cpu().numpy() * base.target_scale
    margin = np.maximum(np.asarray(rows["margin"], dtype=np.float64), 0.0)
    return margin * np.expm1(np.maximum(latent, 0.0))
