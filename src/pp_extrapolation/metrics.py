"""Pooled and physical-unit regression metrics."""
from __future__ import annotations

import numpy as np


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    residual = y - prediction
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else None,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "n": int(len(y)),
    }


def regression_metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    groups = np.asarray(groups)
    pooled = _metrics(y, prediction)
    per_unit = {str(group): _metrics(y[groups == group], prediction[groups == group]) for group in np.unique(groups)}
    finite = [row["r2"] for row in per_unit.values() if row["r2"] is not None]
    return {
        "pooled": pooled,
        "unit_macro_r2": float(np.mean(finite)) if finite else None,
        "per_unit": per_unit,
    }

