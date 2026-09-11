"""Group-OOF, baseline-first convex stacking for PP-X continuations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ContinuousPortfolioDecision:
    weights: tuple[float, ...]
    baseline_weight: float
    total_prior_weight: float
    effective_trust: float
    fold_prior_weights: tuple[float, ...]
    fold_prior_weight_median: float
    fold_prior_weight_iqr: tuple[float, float]
    objective: float


def combine_portfolio(
    expert_predictions: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    predictions = np.asarray(expert_predictions, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if predictions.ndim != 2 or weight.shape != (len(predictions),):
        raise ValueError("predictions must be [expert,row] with one weight each")
    if np.any(weight < 0) or not np.isclose(weight.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must be nonnegative and sum to one")
    return weight @ predictions


def _group_losses(y, groups, prediction):
    labels = np.unique(groups)
    return np.asarray([
        np.mean((prediction[groups == label] - y[groups == label]) ** 2)
        for label in labels
    ])


def _objective(y, groups, experts, weights, rho, tau, gamma, cvar_fraction):
    prediction = weights @ experts
    losses = _group_losses(y, groups, prediction)
    baseline_losses = _group_losses(y, groups, experts[0])
    excess = losses - baseline_losses
    count = max(1, int(np.ceil(cvar_fraction * len(excess))))
    cvar = float(np.mean(np.sort(excess)[-count:]))
    scale = max(float(np.mean(baseline_losses)), 1e-12)
    prior_mass = 1.0 - weights[0]
    return float(
        np.mean(losses) / scale
        + rho * max(cvar, 0.0) / scale
        + tau * prior_mass
        + gamma * np.sum(weights[1:] ** 2)
    )


def _fit_weights(y, groups, experts, rho, tau, gamma, cvar_fraction):
    n = len(experts)
    starts = [np.eye(n)[index] for index in range(n)]
    starts.append(np.full(n, 1.0 / n))
    results = [
        minimize(
            lambda weight: _objective(
                y, groups, experts, weight, rho, tau, gamma, cvar_fraction
            ),
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        for start in starts
    ]
    valid = [result for result in results if result.success]
    if not valid:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")
    result = min(valid, key=lambda item: item.fun)
    weight = np.clip(result.x, 0.0, 1.0)
    return weight / weight.sum()


def select_continuous_portfolio(
    y: np.ndarray,
    groups: np.ndarray,
    expert_predictions: np.ndarray,
    expert_trusts: np.ndarray,
    *,
    rho: float = 0.50,
    tau: float = 0.02,
    gamma: float = 0.01,
    cvar_fraction: float = 0.20,
) -> ContinuousPortfolioDecision:
    """Select stable weights from leave-one-physical-unit-out folds.

    Expert zero must be a fixed trust-zero baseline portfolio. Remaining
    experts are fixed positive-trust portfolios. Each fold selects weights
    without its held-out unit. The coordinate-wise median fold weights are
    used directly; test labels never enter the decision.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    experts = np.asarray(expert_predictions, dtype=np.float64)
    trusts = np.asarray(expert_trusts, dtype=np.float64)
    if y.ndim != 1 or experts.ndim != 2 or experts.shape[1] != len(y):
        raise ValueError("invalid outcome or expert prediction shape")
    if trusts.shape != (len(experts),) or trusts[0] != 0 or np.any(trusts < 0):
        raise ValueError("expert trusts must align and start with zero")
    labels = np.unique(groups)
    if len(labels) < 4:
        raise ValueError("at least four physical units are required")
    if min(rho, tau, gamma) < 0 or not 0 < cvar_fraction <= 1:
        raise ValueError("invalid robust stacking settings")

    fold_weights = []
    for held_out in labels:
        keep = groups != held_out
        fold_weights.append(_fit_weights(
            y[keep], groups[keep], experts[:, keep],
            rho, tau, gamma, cvar_fraction,
        ))
    folds = np.asarray(fold_weights)
    weight = np.median(folds, axis=0)
    if weight.sum() <= 1e-12:
        weight = np.zeros(len(experts))
        weight[0] = 1.0
    else:
        weight /= weight.sum()
    fold_prior = 1.0 - folds[:, 0]
    objective = _objective(
        y, groups, experts, weight, rho, tau, gamma, cvar_fraction
    )
    return ContinuousPortfolioDecision(
        weights=tuple(float(value) for value in weight),
        baseline_weight=float(weight[0]),
        total_prior_weight=float(1.0 - weight[0]),
        effective_trust=float(np.dot(weight, trusts)),
        fold_prior_weights=tuple(float(value) for value in fold_prior),
        fold_prior_weight_median=float(np.median(fold_prior)),
        fold_prior_weight_iqr=tuple(float(value) for value in np.quantile(
            fold_prior, (0.25, 0.75)
        )),
        objective=objective,
    )
