"""Source-only risk-calibrated extension of Innovation Slope Transport.

The extension deliberately avoids target-cohort covariates.  It balances lifecycle
regions during fitting and applies a conservative affine correction only when
cross-fitted source validation improves the worst lifecycle-region error.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any

import numpy as np

from .innovation_slope_transport import fit_innovation_slope_transport as _fit_base
from .innovation_slope_transport import predict_innovation_slope_transport as _predict_base


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _extract_data(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
        def unpack(split: dict[str, Any]) -> tuple[Any, Any]:
            x = next((split[key] for key in ("X", "x", "features") if key in split), None)
            y = next((split[key] for key in ("y", "Y", "target", "targets") if key in split), None)
            if x is None or y is None:
                raise TypeError("SRCIST split dictionaries require feature and target arrays")
            return x, y
        x_train, y_train = unpack(args[0])
        x_val, y_val = unpack(args[1])
        return x_train, y_train, x_val, y_val
    if len(args) >= 4:
        return args[0], args[1], args[2], args[3]
    aliases = (
        ("x_train", "X_train"),
        ("y_train", "Y_train"),
        ("x_val", "X_val", "x_validation", "X_validation"),
        ("y_val", "Y_val", "y_validation", "Y_validation"),
    )
    values = []
    for names in aliases:
        values.append(next((kwargs[name] for name in names if name in kwargs), None))
    if any(value is None for value in values):
        raise TypeError("SRCIST requires explicit train and validation arrays")
    return values[0], values[1], values[2], values[3]


def _balanced_lifecycle_weights(y: Any, strength: float = 0.25) -> np.ndarray:
    target = _as_numpy(y).reshape(-1)
    n = target.size
    order = np.argsort(target, kind="mergesort")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    bins = np.minimum(4, (5 * ranks) // max(n, 1))
    counts = np.bincount(bins, minlength=5).astype(float)
    balanced = n / (5.0 * counts[bins])
    weights = (1.0 - strength) + strength * balanced
    return weights / np.mean(weights)


def _predict_mean(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "mean"):
        pred = model.mean(x)
    elif hasattr(model, "predict"):
        pred = model.predict(x)
    else:
        pred = model(x)
    return _as_numpy(pred)


def _unwrap_model(result: Any) -> Any:
    return result.model if hasattr(result, "model") else (result[0] if isinstance(result, tuple) else result)


def _robust_affine(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = pred.reshape(-1)
    target = y.reshape(-1)
    x0 = float(np.median(x))
    y_scale = max(float(np.std(target)), 1e-8)
    design = np.column_stack((np.ones_like(x), x - x0))
    weight = np.ones_like(x)
    coef = np.array([float(np.median(target)), 1.0])
    ridge = np.diag([1e-8, 1e-3])
    for _ in range(8):
        lhs = design.T @ (weight[:, None] * design) + ridge
        rhs = design.T @ (weight * target) + ridge @ np.array([y_scale * 0.0, 1.0])
        coef = np.linalg.solve(lhs, rhs)
        residual = target - design @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-8)
        weight = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-8))
    slope = float(np.clip(coef[1], 0.65, 1.35))
    center_prediction = float(np.clip(coef[0], np.median(target) - 0.15 * y_scale, np.median(target) + 0.15 * y_scale))
    intercept = center_prediction - slope * x0
    return intercept, slope


def _region_score(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    target = y.reshape(-1)
    estimate = pred.reshape(-1)
    order = np.argsort(target, kind="mergesort")
    bins = np.empty(target.size, dtype=int)
    bins[order] = np.minimum(3, (4 * np.arange(target.size)) // max(target.size, 1))
    losses = [float(np.mean((target[bins == idx] - estimate[bins == idx]) ** 2)) for idx in range(4)]
    return float(np.mean((target - estimate) ** 2)), max(losses)


def _crossfit_calibration(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    raw = pred.reshape(-1)
    target = y.reshape(-1)
    if raw.size < 12 or float(np.std(raw)) < 1e-8:
        return 0.0, 0.0, 1.0
    order = np.argsort(raw, kind="mergesort")
    fold = np.empty(raw.size, dtype=int)
    fold[order] = np.arange(raw.size) % 3
    oof_affine = np.empty_like(raw)
    for idx in range(3):
        train = fold != idx
        intercept, slope = _robust_affine(raw[train], target[train])
        oof_affine[~train] = intercept + slope * raw[~train]
    raw_mean, raw_worst = _region_score(target, raw)
    best_alpha = 0.0
    best_score = raw_worst + 0.20 * raw_mean
    for alpha in (0.25, 0.50, 0.75, 1.0):
        candidate = raw + alpha * (oof_affine - raw)
        mean_loss, worst_loss = _region_score(target, candidate)
        score = worst_loss + 0.20 * mean_loss
        if mean_loss <= 1.01 * raw_mean and score < best_score:
            best_alpha = alpha
            best_score = score
    intercept, slope = _robust_affine(raw, target)
    return best_alpha, intercept, slope


class SourceRiskCalibratedModel:
    def __init__(self, base: Any, alpha: float, intercept: float, slope: float):
        self.base = base
        self.calibration_alpha_ = alpha
        self.calibration_intercept_ = intercept
        self.calibration_slope_ = slope

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _adjust(self, raw: np.ndarray) -> np.ndarray:
        calibrated = self.calibration_intercept_ + self.calibration_slope_ * raw
        return raw + self.calibration_alpha_ * (calibrated - raw)

    def mean(self, x: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        if hasattr(self.base, "mean"):
            raw = _as_numpy(self.base.mean(x, *args, **kwargs))
        else:
            raw = _as_numpy(self.base.predict(x, *args, **kwargs))
        return self._adjust(raw)

    def predict(self, x: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        if hasattr(self.base, "predict"):
            raw = _as_numpy(self.base.predict(x, *args, **kwargs))
        else:
            raw = _as_numpy(self.base.mean(x, *args, **kwargs))
        return self._adjust(raw)


def _rewrap(result: Any, model: Any) -> Any:
    if hasattr(result, "model"):
        try:
            result.model = model
            return result
        except (AttributeError, TypeError):
            if is_dataclass(result):
                return replace(result, model=model)
    if isinstance(result, tuple):
        values = (model,) + tuple(result[1:])
        if hasattr(result, "_fields"):
            return type(result)(*values)
        return values
    return model


def fit_innovation_source_robust_transport(*args: Any, **kwargs: Any) -> Any:
    """Fit CIST with source-only lifecycle balancing and guarded calibration."""
    _, y_train, x_val, y_val = _extract_data(args, kwargs)
    call_kwargs = dict(kwargs)
    existing = call_kwargs.get("sample_weight")
    balanced = _balanced_lifecycle_weights(y_train)
    if existing is not None:
        balanced *= _as_numpy(existing).reshape(-1)
        balanced /= np.mean(balanced)
    call_kwargs["sample_weight"] = balanced
    result = _fit_base(*args, **call_kwargs)
    base_model = _unwrap_model(result)
    val_prediction = _as_numpy(_predict_base(result, x_val))
    alpha, intercept, slope = _crossfit_calibration(val_prediction, _as_numpy(y_val))
    wrapped = SourceRiskCalibratedModel(base_model, alpha, intercept, slope)
    return _rewrap(result, wrapped)


# Compatibility with experiment scripts that retain the original function name.
fit_innovation_slope_transport = fit_innovation_source_robust_transport


def predict_innovation_source_robust_transport(*args: Any, **kwargs: Any) -> Any:
    """Use the established prediction helper with the wrapped SRCIST model."""
    return _predict_base(*args, **kwargs)
