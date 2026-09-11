"""Persistence-anchored residuals invariant to the ordered extrapolation axis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import equal_group_weights
from .risk_budgeted_prior import fit_support_scale, group_mse, support_distance
from .stability_first import regret_summary, unit_regret


@dataclass(frozen=True)
class InvariantResidualFit:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    residual_bound: float
    ridge_alpha: float
    validation_mass: float
    deployment_mass: float
    support: tuple[np.ndarray, np.ndarray, np.ndarray]
    support_threshold: float
    validation_mean_regret: float
    validation_cvar_regret: float
    validation_max_regret: float


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


def fit_invariant_residual(
    train_x,
    train_y,
    train_groups,
    train_anchor,
    validation_x,
    validation_y,
    validation_groups,
    validation_anchor,
    *,
    ridge_alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    mass_grid=np.linspace(0.0, 1.0, 101),
    deployment_fraction=0.25,
    mean_cap=0.02,
    cvar_cap=0.05,
    max_cap=0.10,
) -> InvariantResidualFit:
    """Fit a bounded correction around a causal persistence anchor."""
    train_x = np.asarray(train_x, dtype=np.float64)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    train_anchor = np.asarray(train_anchor, dtype=np.float64)
    validation_anchor = np.asarray(validation_anchor, dtype=np.float64)
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_z = (train_x - center) / scale
    validation_z = (validation_x - center) / scale
    target = train_y - train_anchor
    weights = equal_group_weights(np.asarray(train_groups))
    residual_bound = max(
        float(2.0 * np.quantile(np.abs(target), 0.95)), 1e-8
    )

    candidates = []
    for alpha in ridge_alphas:
        coefficient, intercept = _ridge(
            train_z, target, weights, float(alpha)
        )
        residual = np.clip(
            validation_z @ coefficient + intercept,
            -residual_bound, residual_bound,
        )
        prediction = validation_anchor + residual
        candidates.append((
            float(np.mean(group_mse(
                validation_y, prediction, validation_groups
            ))),
            float(alpha),
            coefficient,
            float(intercept),
            residual,
        ))
    _, alpha, coefficient, intercept, validation_residual = min(
        candidates, key=lambda item: (item[0], item[1])
    )

    feasible = []
    for mass in mass_grid:
        prediction = validation_anchor + float(mass) * validation_residual
        summary = regret_summary(unit_regret(
            validation_y, validation_groups, validation_anchor, prediction
        ))
        if (
            summary[0] <= mean_cap
            and summary[1] <= cvar_cap
            and summary[2] <= max_cap
        ):
            feasible.append((
                float(np.mean(group_mse(
                    validation_y, prediction, validation_groups
                ))),
                float(mass),
            ))
    _, validation_mass = min(feasible, key=lambda item: (item[0], item[1]))
    deployment_mass = float(deployment_fraction * validation_mass)
    deployed_validation = (
        validation_anchor + deployment_mass * validation_residual
    )
    summary = regret_summary(unit_regret(
        validation_y, validation_groups, validation_anchor,
        deployed_validation,
    ))
    support = fit_support_scale(train_x)
    threshold = float(np.max(support_distance(validation_x, support)))
    return InvariantResidualFit(
        center=center,
        scale=scale,
        coefficient=coefficient,
        intercept=intercept,
        residual_bound=residual_bound,
        ridge_alpha=alpha,
        validation_mass=validation_mass,
        deployment_mass=deployment_mass,
        support=support,
        support_threshold=threshold,
        validation_mean_regret=summary[0],
        validation_cvar_regret=summary[1],
        validation_max_regret=summary[2],
    )


def predict_invariant_residual(model, x, anchor):
    """Predict and exactly restore the anchor outside regime support."""
    x = np.asarray(x, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    residual = np.clip(
        ((x - model.center) / model.scale) @ model.coefficient
        + model.intercept,
        -model.residual_bound,
        model.residual_bound,
    )
    out_of_support = support_distance(x, model.support) > model.support_threshold
    prediction = anchor + model.deployment_mass * residual
    return np.where(out_of_support, anchor, prediction), out_of_support
