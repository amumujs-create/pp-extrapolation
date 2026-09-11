"""Risk-certified causal dynamics expert bank for CCMR v2.0."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .model import equal_group_weights
from .risk_budgeted_prior import group_mse
from .stability_first import raw_unit_regret, regret_summary


@dataclass(frozen=True)
class DynamicsExpert:
    name: str
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    fold_coefficients: np.ndarray
    fold_intercepts: np.ndarray


@dataclass(frozen=True)
class CausalDynamicsBankFit:
    experts: tuple[DynamicsExpert, ...]
    expert_weights: np.ndarray
    residual_bound: float
    ridge_alpha: float
    deployment_mass: float
    context_center: np.ndarray
    context_scale: np.ndarray
    context_prototypes: np.ndarray
    support_threshold: float
    sign_threshold: float
    dispersion_threshold: float
    selected_candidate: str
    validation_macro_improvement: float
    validation_mean_regret: float
    validation_cvar_regret: float
    validation_max_regret: float
    validation_active_fraction: float


def _weighted_ridge(x, y, weights, alpha):
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


def _basis(name, correction, center, scale):
    z = (np.asarray(correction, dtype=float) - center) / scale
    linear = z[:, :3]
    if name == "linear_rate":
        return linear
    if name == "damped_acceleration":
        return np.column_stack([
            z[:, 0],
            z[:, 1],
            np.tanh(z[:, 2]),
            z[:, 0] / (1.0 + np.abs(z[:, 0])),
        ])
    if name == "monotone_hinge":
        rate = z[:, 0]
        return np.column_stack([
            rate,
            np.maximum(rate + 1.0, 0.0),
            np.maximum(rate, 0.0),
            np.maximum(rate - 1.0, 0.0),
            z[:, 1],
        ])
    if name == "nonlinear_residual":
        clipped = np.clip(linear, -4.0, 4.0)
        return np.column_stack([
            linear,
            clipped**2,
            clipped[:, 0] * clipped[:, 1],
            clipped[:, 0] * clipped[:, 2],
            np.tanh(clipped),
        ])
    raise ValueError(f"unknown dynamics expert: {name}")


def _fit_expert(name, correction, target, groups, alpha, max_folds):
    center = np.median(correction, axis=0)
    scale = np.quantile(correction, 0.75, axis=0) - np.quantile(
        correction, 0.25, axis=0
    )
    scale = np.where(scale > 1e-12, scale, np.std(correction, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = _basis(name, correction, center, scale)
    weights = equal_group_weights(groups)
    coefficient, intercept = _weighted_ridge(
        design, target, weights, alpha
    )
    labels = np.unique(groups)
    if len(labels) <= max_folds:
        held_out_sets = [np.asarray([label]) for label in labels]
    else:
        held_out_sets = [
            labels[index::max_folds] for index in range(max_folds)
        ]
    fold_coefficients, fold_intercepts = [], []
    for held_out in held_out_sets:
        keep = ~np.isin(groups, held_out)
        fold_coefficient, fold_intercept = _weighted_ridge(
            design[keep],
            target[keep],
            equal_group_weights(groups[keep]),
            alpha,
        )
        fold_coefficients.append(fold_coefficient)
        fold_intercepts.append(fold_intercept)
    return DynamicsExpert(
        name=name,
        center=center,
        scale=scale,
        coefficient=coefficient,
        intercept=float(intercept),
        fold_coefficients=np.asarray(fold_coefficients),
        fold_intercepts=np.asarray(fold_intercepts),
    )


def _expert_predictions(expert, correction):
    design = _basis(
        expert.name, correction, expert.center, expert.scale
    )
    full = design @ expert.coefficient + expert.intercept
    folds = (
        design @ expert.fold_coefficients.T
        + expert.fold_intercepts[None, :]
    ).T
    return full, folds


def _context_fit(context, groups, max_per_group=32):
    center = np.median(context, axis=0)
    scale = np.quantile(context, 0.75, axis=0) - np.quantile(
        context, 0.25, axis=0
    )
    scale = np.where(scale > 1e-12, scale, np.std(context, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (context - center) / scale
    prototypes = []
    for label in np.unique(groups):
        rows = np.flatnonzero(groups == label)
        if len(rows) > max_per_group:
            positions = np.linspace(0, len(rows) - 1, max_per_group)
            rows = rows[np.round(positions).astype(int)]
        prototypes.append(standardized[rows])
    return center, scale, np.concatenate(prototypes)


def _distance(context, center, scale, prototypes):
    standardized = (context - center) / scale
    return cKDTree(prototypes).query(
        standardized, k=1
    )[0] / np.sqrt(standardized.shape[1])


def _consensus(folds, sign_threshold, dispersion_threshold, bound):
    median = np.median(folds, axis=0)
    agreement = np.maximum(
        np.mean(folds > 0, axis=0), np.mean(folds < 0, axis=0)
    )
    dispersion = np.median(
        np.abs(folds - median[None, :]), axis=0
    ) / (np.abs(median) + max(bound * 0.01, 1e-12))
    return (
        np.clip(median, -bound, bound),
        (agreement >= sign_threshold)
        & (dispersion <= dispersion_threshold)
        & (np.abs(median) > 1e-12),
    )


def _weight_candidates(count):
    candidates = []
    for index in range(count):
        value = np.zeros(count)
        value[index] = 1.0
        candidates.append((f"expert_{index}", value))
    for left in range(count):
        for right in range(left + 1, count):
            value = np.zeros(count)
            value[left] = value[right] = 0.5
            candidates.append((f"experts_{left}_{right}", value))
    candidates.append(("uniform_bank", np.ones(count) / count))
    return candidates


def fit_causal_dynamics_bank(
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
    ridge_alphas=(0.1, 1.0, 10.0, 100.0),
    mass_grid=np.linspace(0.0, 1.0, 101),
    sign_threshold=0.90,
    dispersion_threshold=0.50,
    support_quantile=0.99,
    validation_mean_cap=0.0,
    validation_cvar_cap=0.01,
    validation_max_cap=0.02,
    max_ensemble_folds=20,
    expert_names=None,
):
    """Fit an expert-bank residual under physical-unit risk constraints."""
    train_correction_x = np.asarray(train_correction_x, dtype=float)
    validation_correction_x = np.asarray(
        validation_correction_x, dtype=float
    )
    train_context_x = np.asarray(train_context_x, dtype=float)
    validation_context_x = np.asarray(
        validation_context_x, dtype=float
    )
    train_y = np.asarray(train_y, dtype=float)
    validation_y = np.asarray(validation_y, dtype=float)
    train_groups = np.asarray(train_groups)
    validation_groups = np.asarray(validation_groups)
    train_anchor = np.asarray(train_anchor, dtype=float)
    validation_anchor = np.asarray(validation_anchor, dtype=float)
    if len(np.unique(train_groups)) < 3:
        raise ValueError("at least three train groups are required")
    target = train_y - train_anchor
    residual_bound = max(
        float(2.0 * np.quantile(np.abs(target), 0.95)), 1e-8
    )
    available_names = (
        "linear_rate",
        "damped_acceleration",
        "monotone_hinge",
        "nonlinear_residual",
    )
    names = available_names if expert_names is None else tuple(expert_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("expert_names must be unique and nonempty")
    unknown = sorted(set(names) - set(available_names))
    if unknown:
        raise ValueError(f"unknown dynamics experts: {unknown}")
    alpha_scores = []
    for alpha in ridge_alphas:
        experts = tuple(
            _fit_expert(
                name,
                train_correction_x,
                target,
                train_groups,
                float(alpha),
                max_ensemble_folds,
            )
            for name in names
        )
        macro = []
        for expert in experts:
            residual, _ = _expert_predictions(
                expert, validation_correction_x
            )
            prediction = validation_anchor + np.clip(
                residual, -residual_bound, residual_bound
            )
            macro.append(float(np.mean(group_mse(
                validation_y, prediction, validation_groups
            ))))
        alpha_scores.append((float(np.mean(macro)), float(alpha)))
    _, selected_alpha = min(alpha_scores)
    experts = tuple(
        _fit_expert(
            name,
            train_correction_x,
            target,
            train_groups,
            selected_alpha,
            max_ensemble_folds,
        )
        for name in names
    )
    full, folds = zip(*[
        _expert_predictions(expert, validation_correction_x)
        for expert in experts
    ])
    full = np.asarray(full)
    folds = np.asarray(folds)
    context_center, context_scale, prototypes = _context_fit(
        train_context_x, train_groups
    )
    validation_distance = _distance(
        validation_context_x,
        context_center,
        context_scale,
        prototypes,
    )
    support_threshold = float(np.quantile(
        validation_distance, support_quantile
    ))
    baseline_macro = float(np.mean(group_mse(
        validation_y, validation_anchor, validation_groups
    )))
    feasible = []
    for candidate_name, expert_weights in _weight_candidates(len(experts)):
        mixed_full = expert_weights @ full
        mixed_folds = np.tensordot(
            expert_weights, folds, axes=(0, 0)
        )
        _, consensus = _consensus(
            mixed_folds,
            sign_threshold,
            dispersion_threshold,
            residual_bound,
        )
        active = consensus & (
            validation_distance <= support_threshold
        )
        residual = np.where(
            active,
            np.clip(mixed_full, -residual_bound, residual_bound),
            0.0,
        )
        for mass in mass_grid:
            prediction = (
                validation_anchor + float(mass) * residual
            )
            summary = regret_summary(raw_unit_regret(
                validation_y,
                validation_groups,
                validation_anchor,
                prediction,
            ))
            macro = float(np.mean(group_mse(
                validation_y, prediction, validation_groups
            )))
            if (
                summary[0] <= validation_mean_cap + 1e-12
                and summary[1] <= validation_cvar_cap + 1e-12
                and summary[2] <= validation_max_cap + 1e-12
            ):
                feasible.append((
                    macro,
                    np.count_nonzero(expert_weights),
                    float(mass),
                    candidate_name,
                    expert_weights.copy(),
                    float(np.mean(active)),
                    *summary,
                ))
    chosen = min(feasible, key=lambda row: (row[0], row[1], row[2]))
    (
        macro,
        _,
        deployment_mass,
        candidate_name,
        expert_weights,
        active_fraction,
        mean_regret,
        cvar_regret,
        max_regret,
    ) = chosen
    if macro >= baseline_macro - 1e-12:
        deployment_mass = 0.0
        macro = baseline_macro
        mean_regret = cvar_regret = max_regret = 0.0
    return CausalDynamicsBankFit(
        experts=experts,
        expert_weights=expert_weights,
        residual_bound=residual_bound,
        ridge_alpha=selected_alpha,
        deployment_mass=deployment_mass,
        context_center=context_center,
        context_scale=context_scale,
        context_prototypes=prototypes,
        support_threshold=support_threshold,
        sign_threshold=sign_threshold,
        dispersion_threshold=dispersion_threshold,
        selected_candidate=candidate_name,
        validation_macro_improvement=float(
            (baseline_macro - macro) / max(baseline_macro, 1e-12)
        ),
        validation_mean_regret=mean_regret,
        validation_cvar_regret=cvar_regret,
        validation_max_regret=max_regret,
        validation_active_fraction=active_fraction,
    )


def predict_causal_dynamics_bank(model, correction_x, context_x, anchor):
    """Predict with exact persistence fallback outside certified support."""
    correction_x = np.asarray(correction_x, dtype=float)
    context_x = np.asarray(context_x, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    predictions = [
        _expert_predictions(expert, correction_x)
        for expert in model.experts
    ]
    full = np.asarray([value[0] for value in predictions])
    folds = np.asarray([value[1] for value in predictions])
    mixed_full = model.expert_weights @ full
    mixed_folds = np.tensordot(
        model.expert_weights, folds, axes=(0, 0)
    )
    _, consensus = _consensus(
        mixed_folds,
        model.sign_threshold,
        model.dispersion_threshold,
        model.residual_bound,
    )
    distance = _distance(
        context_x,
        model.context_center,
        model.context_scale,
        model.context_prototypes,
    )
    support = distance <= model.support_threshold
    active = consensus & support & (model.deployment_mass > 0)
    residual = np.clip(
        mixed_full, -model.residual_bound, model.residual_bound
    )
    prediction = anchor + np.where(
        active, model.deployment_mass * residual, 0.0
    )
    return prediction, {
        "active": active,
        "support_rejected": ~support,
        "consensus_rejected": ~consensus,
        "distance": distance,
    }
