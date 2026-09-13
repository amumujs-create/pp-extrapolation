"""Environment-discovered distributionally robust slope transport (EDIST).

EDIST discovers latent operating regimes using source features only. A pilot CIST
estimates source-validation risk in each regime, then a second CIST is fitted with
bounded group-DRO weights. The pilot/final consensus is selected by worst-regime
source risk subject to a pooled-risk safeguard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .innovation_slope_transport import (
    fit_innovation_slope_transport as _fit_cist,
    predict_innovation_slope_transport as _predict_cist,
)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _split_arrays(split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = next((_array(split[key]) for key in ("X", "x", "features") if key in split), None)
    y = next((_array(split[key]).reshape(-1) for key in ("y", "Y", "target", "targets") if key in split), None)
    if x is None or y is None:
        raise KeyError("EDIST requires feature and target arrays")
    return x, y


def _latent_coordinates(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    progress_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = [idx for idx in range(train_x.shape[1]) if idx != progress_index]
    if not keep:
        keep = [progress_index]
    train = np.asarray(train_x[:, keep], dtype=float)
    validation = np.asarray(validation_x[:, keep], dtype=float)
    median = np.nanmedian(train, axis=0)
    train = np.where(np.isfinite(train), train, median)
    validation = np.where(np.isfinite(validation), validation, median)
    mad = 1.4826 * np.nanmedian(np.abs(train - median), axis=0)
    std = np.nanstd(train, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    train = (train - median) / scale
    validation = (validation - median) / scale
    _, _, vt = np.linalg.svd(train, full_matrices=False)
    components = vt[: max(1, min(5, vt.shape[0]))].T
    return train @ components, validation @ components


def _discover_environments(
    train_z: np.ndarray,
    validation_z: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Quantile initialization on the first source principal coordinate is stable
    # across seeds and prevents environment discovery from adding random variance.
    order = np.argsort(train_z[:, 0], kind="mergesort")
    anchors = np.linspace(0, len(order) - 1, n_groups).round().astype(int)
    centers = train_z[order[anchors]].copy()
    train_group = np.zeros(train_z.shape[0], dtype=int)
    for _ in range(30):
        distance = np.sum((train_z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated_group = np.argmin(distance, axis=1)
        updated_centers = centers.copy()
        for group in range(n_groups):
            members = train_z[updated_group == group]
            if members.size:
                updated_centers[group] = np.mean(members, axis=0)
        if np.array_equal(updated_group, train_group) and np.allclose(updated_centers, centers):
            train_group = updated_group
            centers = updated_centers
            break
        train_group = updated_group
        centers = updated_centers
    validation_distance = np.sum((validation_z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return train_group, np.argmin(validation_distance, axis=1)


def _group_risks(y: np.ndarray, prediction: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    residual2 = (y.reshape(-1) - prediction.reshape(-1)) ** 2
    pooled = max(float(np.mean(residual2)), 1e-8)
    risks = np.full(n_groups, pooled)
    for idx in range(n_groups):
        members = group == idx
        if np.any(members):
            risks[idx] = float(np.mean(residual2[members]))
    return risks


def _dro_sample_weights(train_group: np.ndarray, validation_risk: np.ndarray) -> np.ndarray:
    counts = np.bincount(train_group, minlength=validation_risk.size).astype(float)
    relative_risk = validation_risk / max(float(np.mean(validation_risk)), 1e-8)
    adversarial = np.exp(np.clip((relative_risk - 1.0) / 0.75, -1.2, 1.2))
    equal_group = train_group.size / np.maximum(validation_risk.size * counts, 1.0)
    raw = adversarial[train_group] * equal_group[train_group]
    raw /= max(float(np.mean(raw)), 1e-8)
    bounded = np.clip(raw, 0.45, 2.50)
    # Retain half of ordinary empirical-risk training for extrapolation safety.
    weights = 0.50 + 0.50 * bounded
    return weights / np.mean(weights)


def _select_consensus(
    y: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    pilot: np.ndarray,
    dro: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    candidates = (0.0, 0.25, 0.50, 0.75, 1.0)
    records = []
    for weight in candidates:
        prediction = pilot + weight * (dro - pilot)
        risks = _group_risks(y, prediction, group, n_groups)
        pooled = float(np.mean((y - prediction) ** 2))
        objective = float(np.max(risks) + 0.15 * pooled + 0.10 * np.std(risks))
        records.append((objective, pooled, weight, float(np.max(risks)), float(np.std(risks))))
    component_best = min(records[0][1], records[-1][1])
    admissible = [record for record in records if record[1] <= 1.01 * component_best]
    selected = min(admissible or records, key=lambda record: (record[0], abs(record[2] - 0.5)))
    objective, pooled, weight, worst, dispersion = selected
    return weight, pooled, {
        "worst_environment_mse": worst,
        "environment_risk_sd": dispersion,
        "gate_objective": objective,
        "pilot_validation_mse": records[0][1],
        "dro_validation_mse": records[-1][1],
    }


@dataclass
class EnvironmentDROFit:
    pilot_fit: Any
    dro_fit: Any
    dro_weight: float
    validation_mse: float
    selected_epoch: int
    environment_risks: list[float]
    diagnostics: dict[str, float]


def fit_environment_dro_slope_transport(
    train: dict[str, Any],
    validation: dict[str, Any],
    *,
    progress_index: int,
    **kwargs: Any,
) -> EnvironmentDROFit:
    train_x, _ = _split_arrays(train)
    validation_x, validation_y = _split_arrays(validation)
    n_groups = max(2, min(4, train_x.shape[0] // 30, validation_x.shape[0] // 8))
    train_z, validation_z = _latent_coordinates(train_x, validation_x, progress_index)
    train_group, validation_group = _discover_environments(train_z, validation_z, n_groups)

    pilot_fit = _fit_cist(train, validation, progress_index=progress_index, **kwargs)
    pilot_validation = _array(_predict_cist(pilot_fit, validation_x)).reshape(-1)
    risks = _group_risks(validation_y, pilot_validation, validation_group, n_groups)
    sample_weight = _dro_sample_weights(train_group, risks)
    existing_weight = kwargs.pop("sample_weight", None)
    if existing_weight is not None:
        sample_weight *= _array(existing_weight).reshape(-1)
        sample_weight /= np.mean(sample_weight)
    dro_fit = _fit_cist(
        train,
        validation,
        progress_index=progress_index,
        sample_weight=sample_weight,
        **kwargs,
    )
    dro_validation = _array(_predict_cist(dro_fit, validation_x)).reshape(-1)
    weight, validation_mse, diagnostics = _select_consensus(
        validation_y,
        validation_group,
        n_groups,
        pilot_validation,
        dro_validation,
    )
    pilot_epoch = int(getattr(pilot_fit, "selected_epoch", 0))
    dro_epoch = int(getattr(dro_fit, "selected_epoch", 0))
    selected_epoch = int(round((1.0 - weight) * pilot_epoch + weight * dro_epoch))
    return EnvironmentDROFit(
        pilot_fit=pilot_fit,
        dro_fit=dro_fit,
        dro_weight=weight,
        validation_mse=validation_mse,
        selected_epoch=selected_epoch,
        environment_risks=risks.tolist(),
        diagnostics=diagnostics,
    )


def predict_environment_dro_slope_transport(
    fit: EnvironmentDROFit,
    x: Any,
    **kwargs: Any,
) -> np.ndarray:
    pilot = _array(_predict_cist(fit.pilot_fit, x, **kwargs))
    dro = _array(_predict_cist(fit.dro_fit, x, **kwargs))
    return pilot + fit.dro_weight * (dro - pilot)

