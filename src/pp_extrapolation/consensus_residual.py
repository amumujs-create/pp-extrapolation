"""Cross-fit consensus and minimax-certified residual correction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .model import equal_group_weights
from .risk_budgeted_prior import group_mse
from .stability_first import raw_unit_regret, regret_summary


@dataclass(frozen=True)
class ConsensusResidualFit:
    correction_center: np.ndarray
    correction_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    coefficients: np.ndarray
    intercepts: np.ndarray
    residual_bound: float
    ridge_alpha: float
    ensemble_folds: int
    deployment_mass: float
    context_center: np.ndarray
    context_scale: np.ndarray
    context_prototypes: np.ndarray
    support_threshold: float
    sign_threshold: float
    dispersion_threshold: float
    validation_macro_improvement: float
    validation_mean_regret: float
    validation_cvar_regret: float
    validation_max_regret: float
    validation_active_fraction: float


def _ridge(x, y, weights, alpha):
    total = float(np.sum(weights))
    x_mean = np.sum(weights[:, None] * x, axis=0) / total
    y_mean = float(np.sum(weights * y) / total)
    centered_x = x - x_mean
    centered_y = y - y_mean
    coefficient = np.linalg.solve(
        centered_x.T @ (weights[:, None] * centered_x)
        + float(alpha) * np.eye(x.shape[1]),
        centered_x.T @ (weights * centered_y),
    )
    return coefficient, y_mean - float(x_mean @ coefficient)


def _ensemble_fit(x, target, groups, alpha, max_folds=None):
    coefficients, intercepts = [], []
    labels = np.unique(groups)
    if max_folds is None or len(labels) <= int(max_folds):
        held_out_sets = [np.asarray([label]) for label in labels]
    else:
        held_out_sets = [
            labels[index::int(max_folds)]
            for index in range(int(max_folds))
        ]
    for held_out in held_out_sets:
        keep = ~np.isin(groups, held_out)
        coefficient, intercept = _ridge(
            x[keep], target[keep], equal_group_weights(groups[keep]), alpha
        )
        coefficients.append(coefficient)
        intercepts.append(intercept)
    return np.asarray(coefficients), np.asarray(intercepts)


def _consensus(ensemble, sign_threshold, dispersion_threshold, bound):
    median = np.median(ensemble, axis=0)
    positive = np.mean(ensemble > 0, axis=0)
    negative = np.mean(ensemble < 0, axis=0)
    sign_agreement = np.maximum(positive, negative)
    mad = np.median(np.abs(ensemble - median[None, :]), axis=0)
    relative_dispersion = mad / (
        np.abs(median) + max(float(bound) * 0.01, 1e-12)
    )
    active = (
        (sign_agreement >= sign_threshold)
        & (relative_dispersion <= dispersion_threshold)
        & (np.abs(median) > 1e-12)
    )
    return np.clip(median, -bound, bound), active


def _context_fit(context, groups, max_per_group=32):
    center = np.median(context, axis=0)
    scale = np.quantile(context, 0.75, axis=0) - np.quantile(
        context, 0.25, axis=0
    )
    fallback = np.std(context, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (context - center) / scale
    selected = []
    for label in np.unique(groups):
        rows = np.flatnonzero(groups == label)
        if len(rows) > max_per_group:
            positions = np.linspace(0, len(rows) - 1, max_per_group)
            rows = rows[np.round(positions).astype(int)]
        selected.append(z[rows])
    return center, scale, np.concatenate(selected)


def _nearest_distance(context, center, scale, prototypes):
    z = (context - center) / scale
    return cKDTree(prototypes).query(z, k=1)[0] / np.sqrt(z.shape[1])


def fit_consensus_residual(
    train_correction_x,
    train_context_x,
    train_y,
    train_groups,
    train_anchor,
    validation_correction_x,
    validation_context_x,
    validation_y,
    validation_groups,
    validation_anchor,
    *,
    ridge_alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    mass_grid=np.linspace(0.0, 0.25, 101),
    sign_threshold=0.90,
    dispersion_threshold=0.50,
    support_quantile=0.99,
    validation_mean_cap=0.0,
    validation_cvar_cap=0.01,
    validation_max_cap=0.02,
    max_ensemble_folds=None,
) -> ConsensusResidualFit:
    """Fit a correction accepted only by fold consensus and minimax validation."""
    train_correction_x = np.asarray(train_correction_x, dtype=np.float64)
    validation_correction_x = np.asarray(
        validation_correction_x, dtype=np.float64
    )
    train_context_x = np.asarray(train_context_x, dtype=np.float64)
    validation_context_x = np.asarray(validation_context_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    train_groups = np.asarray(train_groups)
    validation_groups = np.asarray(validation_groups)
    train_anchor = np.asarray(train_anchor, dtype=np.float64)
    validation_anchor = np.asarray(validation_anchor, dtype=np.float64)
    if len(np.unique(train_groups)) < 3:
        raise ValueError("at least three train groups are required")

    center = np.mean(train_correction_x, axis=0)
    scale = np.std(train_correction_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_z = (train_correction_x - center) / scale
    validation_z = (validation_correction_x - center) / scale
    target = train_y - train_anchor
    residual_bound = max(
        float(2.0 * np.quantile(np.abs(target), 0.95)), 1e-8
    )
    weights = equal_group_weights(train_groups)
    candidates = []
    for alpha in ridge_alphas:
        coefficient, intercept = _ridge(
            train_z, target, weights, float(alpha)
        )
        prediction = validation_anchor + np.clip(
            validation_z @ coefficient + intercept,
            -residual_bound,
            residual_bound,
        )
        candidates.append((
            float(np.mean(group_mse(
                validation_y, prediction, validation_groups
            ))),
            float(alpha),
        ))
    _, ridge_alpha = min(candidates, key=lambda row: (row[0], row[1]))
    coefficient, intercept = _ridge(
        train_z, target, weights, ridge_alpha
    )
    coefficients, intercepts = _ensemble_fit(
        train_z,
        target,
        train_groups,
        ridge_alpha,
        max_folds=max_ensemble_folds,
    )
    validation_ensemble = (
        validation_z @ coefficients.T + intercepts[None, :]
    ).T
    _, consensus = _consensus(
        validation_ensemble,
        float(sign_threshold),
        float(dispersion_threshold),
        residual_bound,
    )
    validation_residual = np.clip(
        validation_z @ coefficient + intercept,
        -residual_bound,
        residual_bound,
    )

    context_center, context_scale, prototypes = _context_fit(
        train_context_x, train_groups
    )
    validation_distance = _nearest_distance(
        validation_context_x,
        context_center,
        context_scale,
        prototypes,
    )
    support_threshold = float(np.quantile(
        validation_distance, support_quantile
    ))
    active = consensus & (validation_distance <= support_threshold)
    gated_residual = np.where(active, validation_residual, 0.0)
    baseline_macro = float(np.mean(group_mse(
        validation_y, validation_anchor, validation_groups
    )))
    feasible = []
    for mass in mass_grid:
        prediction = validation_anchor + float(mass) * gated_residual
        per_group_regret = raw_unit_regret(
            validation_y,
            validation_groups,
            validation_anchor,
            prediction,
        )
        summary = regret_summary(per_group_regret)
        macro = float(np.mean(group_mse(
            validation_y, prediction, validation_groups
        )))
        if (
            summary[0] <= validation_mean_cap + 1e-12
            and summary[1] <= validation_cvar_cap + 1e-12
            and summary[2] <= validation_max_cap + 1e-12
        ):
            feasible.append((macro, float(mass), *summary))
    macro, deployment_mass, mean_regret, cvar_regret, max_regret = min(
        feasible, key=lambda row: (row[0], row[1])
    )
    if macro >= baseline_macro - 1e-12:
        deployment_mass = 0.0
        macro = baseline_macro
        mean_regret = cvar_regret = max_regret = 0.0
    return ConsensusResidualFit(
        correction_center=center,
        correction_scale=scale,
        coefficient=coefficient,
        intercept=float(intercept),
        coefficients=coefficients,
        intercepts=intercepts,
        residual_bound=residual_bound,
        ridge_alpha=ridge_alpha,
        ensemble_folds=len(intercepts),
        deployment_mass=deployment_mass,
        context_center=context_center,
        context_scale=context_scale,
        context_prototypes=prototypes,
        support_threshold=support_threshold,
        sign_threshold=float(sign_threshold),
        dispersion_threshold=float(dispersion_threshold),
        validation_macro_improvement=float(
            (baseline_macro - macro) / max(baseline_macro, 1e-12)
        ),
        validation_mean_regret=mean_regret,
        validation_cvar_regret=cvar_regret,
        validation_max_regret=max_regret,
        validation_active_fraction=float(np.mean(active)),
    )


def predict_consensus_residual(
    model,
    correction_x,
    context_x,
    anchor,
):
    """Predict with exact anchor fallback outside consensus/context support."""
    correction_x = np.asarray(correction_x, dtype=np.float64)
    context_x = np.asarray(context_x, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    z = (
        (correction_x - model.correction_center)
        / model.correction_scale
    )
    ensemble = (z @ model.coefficients.T + model.intercepts[None, :]).T
    _, consensus = _consensus(
        ensemble,
        model.sign_threshold,
        model.dispersion_threshold,
        model.residual_bound,
    )
    residual = np.clip(
        z @ model.coefficient + model.intercept,
        -model.residual_bound,
        model.residual_bound,
    )
    distance = _nearest_distance(
        context_x,
        model.context_center,
        model.context_scale,
        model.context_prototypes,
    )
    in_support = distance <= model.support_threshold
    active = consensus & in_support & (model.deployment_mass > 0)
    prediction = anchor + model.deployment_mass * np.where(
        active, residual, 0.0
    )
    return prediction, {
        "active": active,
        "consensus_rejected": ~consensus,
        "support_rejected": ~in_support,
        "support_distance": distance,
    }
