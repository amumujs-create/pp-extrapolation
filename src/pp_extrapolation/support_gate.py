"""Support-aware decay for PP's nonlinear correction path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .hull import ConvexHullAudit, audit_convex_hull_support
from .model import PPFit, transform_features


DEFAULT_BETAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class SupportGateSelection:
    """Validation-only selection result for the support decay strength."""

    beta: float
    candidates: tuple[dict, ...]


def support_distance(
    train_coordinate: np.ndarray,
    source_coordinate: np.ndarray,
) -> tuple[np.ndarray, ConvexHullAudit]:
    """Return standardized distance outside the train convex hull."""
    audit = audit_convex_hull_support(train_coordinate, source_coordinate)
    return audit.distance.astype(np.float64), audit


def predict_components(fit: PPFit, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return affine prediction and nonlinear correction in target units."""
    fit.model.eval()
    value = torch.as_tensor(
        transform_features(x, fit.center, fit.scale), dtype=torch.float32
    )
    with torch.no_grad():
        affine = fit.model.affine(value).squeeze(1).cpu().numpy()
        correction = fit.model.nonlinear(value).squeeze(1).cpu().numpy()
    return (
        affine.astype(np.float64) * fit.target_scale,
        correction.astype(np.float64) * fit.target_scale,
    )


def combine_support_gated(
    affine: np.ndarray,
    correction: np.ndarray,
    distance: np.ndarray,
    *,
    beta: float,
    output_cap: float,
) -> np.ndarray:
    """Decay only the nonlinear path outside support; beta=0 is baseline PP."""
    affine = np.asarray(affine, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if affine.shape != correction.shape or affine.shape != distance.shape:
        raise ValueError("affine, correction, and distance must have identical shapes")
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    gate = np.exp(-float(beta) * np.maximum(distance, 0.0))
    return np.clip(affine + gate * correction, 0.0, float(output_cap))


def select_support_gate(
    fit: PPFit,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_distance: np.ndarray,
    *,
    betas: Iterable[float] = DEFAULT_BETAS,
) -> SupportGateSelection:
    """Choose beta using unit-disjoint validation data only."""
    affine, correction = predict_components(fit, validation_x)
    truth = np.asarray(validation_y, dtype=np.float64)
    rows = []
    for beta in betas:
        prediction = combine_support_gated(
            affine,
            correction,
            validation_distance,
            beta=float(beta),
            output_cap=fit.target_scale,
        )
        rows.append({
            "beta": float(beta),
            "validation_mse": float(np.mean((prediction - truth) ** 2)),
        })
    selected = min(rows, key=lambda row: (row["validation_mse"], row["beta"]))
    return SupportGateSelection(float(selected["beta"]), tuple(rows))


def predict_support_gated(
    fit: PPFit,
    x: np.ndarray,
    distance: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Predict with validation-selected support decay."""
    affine, correction = predict_components(fit, x)
    return combine_support_gated(
        affine, correction, distance, beta=beta, output_cap=fit.target_scale
    )
