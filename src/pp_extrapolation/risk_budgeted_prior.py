"""Baseline-relative empirical risk budgets for prior residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskBudgetDecision:
    alpha: float
    mean_excess_ratio: float
    tail_excess_ratio: float
    feasible_count: int
    epsilon: float
    cvar_fraction: float


def group_mse(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return one equally weighted squared-error loss per physical unit."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    groups = np.asarray(groups)
    if y.ndim != 1 or prediction.shape != y.shape or groups.shape != y.shape:
        raise ValueError("y, prediction, and groups must be aligned vectors")
    return np.asarray([
        np.mean((prediction[groups == label] - y[groups == label]) ** 2)
        for label in np.unique(groups)
    ])


def apply_prior_residual(
    baseline: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    modulation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a bounded, optionally sample-specific prior residual budget."""
    baseline = np.asarray(baseline, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if baseline.shape != prior.shape or baseline.ndim != 1:
        raise ValueError("baseline and prior must be aligned vectors")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if modulation is None:
        modulation = np.ones_like(baseline)
    modulation = np.asarray(modulation, dtype=np.float64)
    if modulation.shape != baseline.shape or np.any(
        (modulation < 0) | (modulation > 1)
    ):
        raise ValueError("modulation must be aligned and in [0, 1]")
    return baseline + alpha * modulation * (prior - baseline)


def _risk_ratios(y, groups, baseline, prediction, cvar_fraction):
    base_loss = group_mse(y, baseline, groups)
    loss = group_mse(y, prediction, groups)
    excess = loss - base_loss
    scale = max(float(np.mean(base_loss)), 1e-12)
    count = max(1, int(np.ceil(cvar_fraction * len(excess))))
    return (
        float(np.mean(excess) / scale),
        float(np.mean(np.sort(excess)[-count:]) / scale),
    )


def select_risk_budget(
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prior: np.ndarray,
    modulation: np.ndarray | None = None,
    *,
    epsilon: float = 0.02,
    cvar_fraction: float = 0.20,
    enforce_tail: bool = True,
    grid_size: int = 101,
) -> RiskBudgetDecision:
    """Choose the largest empirically feasible prior budget on validation.

    Feasibility requires both mean excess group MSE and, by default, upper-tail
    excess group MSE to be no greater than ``epsilon`` times baseline mean
    group MSE. Alpha zero is therefore always an exact baseline fallback.
    """
    if epsilon < 0 or not 0 < cvar_fraction <= 1 or grid_size < 2:
        raise ValueError("invalid risk-budget settings")
    feasible = []
    for alpha in np.linspace(0.0, 1.0, grid_size):
        prediction = apply_prior_residual(baseline, prior, alpha, modulation)
        mean_ratio, tail_ratio = _risk_ratios(
            y, groups, baseline, prediction, cvar_fraction
        )
        if mean_ratio <= epsilon + 1e-12 and (
            not enforce_tail or tail_ratio <= epsilon + 1e-12
        ):
            feasible.append((float(alpha), mean_ratio, tail_ratio))
    alpha, mean_ratio, tail_ratio = feasible[-1]
    return RiskBudgetDecision(
        alpha=alpha,
        mean_excess_ratio=mean_ratio,
        tail_excess_ratio=tail_ratio,
        feasible_count=len(feasible),
        epsilon=float(epsilon),
        cvar_fraction=float(cvar_fraction),
    )


def fit_support_scale(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit train-only axis support bounds and robust feature scales."""
    train_x = np.asarray(train_x, dtype=np.float64)
    if train_x.ndim != 2 or len(train_x) < 2:
        raise ValueError("train_x must be a nonempty matrix")
    lower = np.min(train_x, axis=0)
    upper = np.max(train_x, axis=0)
    scale = np.quantile(train_x, 0.75, axis=0) - np.quantile(
        train_x, 0.25, axis=0
    )
    scale = np.where(scale > 1e-12, scale, np.std(train_x, axis=0))
    return lower, upper, np.where(scale > 1e-12, scale, 1.0)


def support_distance(
    x: np.ndarray, support: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Axis-hull distance in train robust-scale units."""
    x = np.asarray(x, dtype=np.float64)
    lower, upper, scale = support
    if x.ndim != 2 or x.shape[1] != len(lower):
        raise ValueError("x does not align with fitted support")
    outside = np.maximum(np.maximum(lower - x, x - upper), 0.0) / scale
    return np.sqrt(np.mean(outside**2, axis=1))


def local_budget_modulation(
    distance: np.ndarray,
    disagreement: np.ndarray,
    distance_scale: float,
    disagreement_scale: float,
) -> np.ndarray:
    """Monotonically reduce prior mass outside support or under disagreement."""
    distance = np.asarray(distance, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=np.float64)
    if distance.shape != disagreement.shape or distance.ndim != 1:
        raise ValueError("distance and disagreement must be aligned vectors")
    d_scale = max(float(distance_scale), 1e-12)
    u_scale = max(float(disagreement_scale), 1e-12)
    return 1.0 / (1.0 + distance / d_scale + disagreement / u_scale)
