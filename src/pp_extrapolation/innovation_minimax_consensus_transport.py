"""Minimax consensus between structural and source-robust slope transport.

MCST keeps target cohorts completely sealed.  Its gate is selected on source
validation pseudo-cohorts using pooled loss, worst lifecycle-region loss, and
fold instability rather than a single average validation score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .innovation_slope_transport import (
    fit_innovation_slope_transport as _fit_plain,
    predict_innovation_slope_transport as _predict_plain,
)
from .innovation_source_robust_transport import (
    fit_innovation_source_robust_transport as _fit_robust,
    predict_innovation_source_robust_transport as _predict_robust,
)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _target(split: dict[str, Any]) -> np.ndarray:
    for key in ("y", "Y", "target", "targets"):
        if key in split:
            return _array(split[key]).reshape(-1)
    raise KeyError("MCST requires targets in the validation split")


def _features(split: dict[str, Any]) -> Any:
    for key in ("X", "x", "features"):
        if key in split:
            return split[key]
    raise KeyError("MCST requires features in the validation split")


def _candidate_statistics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    residual2 = (y - prediction) ** 2
    order = np.argsort(y, kind="mergesort")
    region = np.empty(y.size, dtype=int)
    region[order] = np.minimum(3, (4 * np.arange(y.size)) // max(y.size, 1))
    region_mse = np.array([np.mean(residual2[region == idx]) for idx in range(4)])

    # Interleave the ordered lifecycle so every pseudo-fold spans the full range.
    fold = np.empty(y.size, dtype=int)
    fold[order] = np.arange(y.size) % 3
    fold_mse = np.array([np.mean(residual2[fold == idx]) for idx in range(3)])
    return float(np.mean(residual2)), float(np.max(region_mse)), float(np.std(fold_mse))


def _select_gate(y: np.ndarray, plain: np.ndarray, robust: np.ndarray) -> tuple[float, float, dict[str, float]]:
    y = y.reshape(-1)
    plain = plain.reshape(-1)
    robust = robust.reshape(-1)
    scale = max(float(np.var(y)), 1e-8)
    candidates = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0, 1.20)
    records: list[tuple[float, float, float, float, float]] = []
    for weight in candidates:
        prediction = plain + weight * (robust - plain)
        pooled, worst, instability = _candidate_statistics(y, prediction)
        objective = pooled + 0.20 * worst + 0.10 * instability + 0.002 * scale * abs(weight - 0.5)
        records.append((objective, pooled, worst, instability, weight))

    plain_pooled = records[0][1]
    robust_pooled = next(row[1] for row in records if row[4] == 1.0)
    admissible = [row for row in records if row[1] <= 1.01 * min(plain_pooled, robust_pooled)]
    selected = min(admissible or records, key=lambda row: (row[0], abs(row[4] - 0.5)))
    objective, pooled, worst, instability, weight = selected
    return weight, pooled, {
        "gate_objective": objective,
        "gate_worst_region_mse": worst,
        "gate_fold_instability": instability,
        "plain_validation_mse": plain_pooled,
        "robust_validation_mse": robust_pooled,
    }


@dataclass
class MinimaxConsensusFit:
    plain_fit: Any
    robust_fit: Any
    robust_weight: float
    validation_mse: float
    selected_epoch: int
    gate_diagnostics: dict[str, float]


def fit_innovation_minimax_consensus_transport(
    train: dict[str, Any],
    validation: dict[str, Any],
    **kwargs: Any,
) -> MinimaxConsensusFit:
    plain_fit = _fit_plain(train, validation, **kwargs)
    robust_fit = _fit_robust(train, validation, **kwargs)
    validation_x = _features(validation)
    plain_prediction = _array(_predict_plain(plain_fit, validation_x)).reshape(-1)
    robust_prediction = _array(_predict_robust(robust_fit, validation_x)).reshape(-1)
    weight, validation_mse, diagnostics = _select_gate(
        _target(validation), plain_prediction, robust_prediction
    )
    plain_epoch = int(getattr(plain_fit, "selected_epoch", 0))
    robust_epoch = int(getattr(robust_fit, "selected_epoch", 0))
    selected_epoch = int(round((1.0 - min(weight, 1.0)) * plain_epoch + min(weight, 1.0) * robust_epoch))
    return MinimaxConsensusFit(
        plain_fit=plain_fit,
        robust_fit=robust_fit,
        robust_weight=weight,
        validation_mse=validation_mse,
        selected_epoch=selected_epoch,
        gate_diagnostics=diagnostics,
    )


def predict_innovation_minimax_consensus_transport(
    fit: MinimaxConsensusFit,
    x: Any,
    **kwargs: Any,
) -> np.ndarray:
    plain = _array(_predict_plain(fit.plain_fit, x, **kwargs))
    robust = _array(_predict_robust(fit.robust_fit, x, **kwargs))
    return plain + fit.robust_weight * (robust - plain)
