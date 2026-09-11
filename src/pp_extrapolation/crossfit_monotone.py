"""Cross-fitted monotone risk budgets for prior residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk_budgeted_prior import apply_prior_residual, group_mse


@dataclass(frozen=True)
class MonotoneBudgetFit:
    alpha: float
    distance_decay: float
    disagreement_decay: float
    validation_loss: float
    mean_excess_ratio: float
    tail_excess_ratio: float


@dataclass(frozen=True)
class CrossfitMonotoneResult:
    final: MonotoneBudgetFit
    fold_parameters: tuple[tuple[float, float, float], ...]
    oof_prediction: np.ndarray
    oof_mean_excess_ratio: float
    oof_tail_excess_ratio: float
    oof_feasible: bool


def monotone_modulation(
    distance: np.ndarray,
    disagreement: np.ndarray,
    distance_decay: float,
    disagreement_decay: float,
) -> np.ndarray:
    """Prior mass decreases monotonically with both risk indicators."""
    distance = np.asarray(distance, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=np.float64)
    if distance.shape != disagreement.shape or distance.ndim != 1:
        raise ValueError("risk indicators must be aligned vectors")
    if min(distance_decay, disagreement_decay) < 0:
        raise ValueError("decays must be nonnegative")
    return np.exp(
        -distance_decay * np.maximum(distance, 0.0)
        - disagreement_decay * np.maximum(disagreement, 0.0)
    )


def risk_ratios(y, groups, baseline, prediction, cvar_fraction=0.20):
    """Raw equally weighted physical-unit mean and CVaR excess ratios."""
    return hierarchical_risk_ratios(
        y, groups, baseline, prediction, cvar_fraction, shrinkage_n0=0.0
    )


def hierarchical_risk_ratios(
    y, groups, baseline, prediction, cvar_fraction=0.20, shrinkage_n0=5.0
):
    """Mean raw excess and partially pooled upper-tail excess ratios."""
    if shrinkage_n0 < 0:
        raise ValueError("shrinkage_n0 must be nonnegative")
    baseline_loss = group_mse(y, baseline, groups)
    loss = group_mse(y, prediction, groups)
    excess = loss - baseline_loss
    scale = max(float(np.mean(baseline_loss)), 1e-12)
    count = max(1, int(np.ceil(cvar_fraction * len(excess))))
    labels, counts = np.unique(groups, return_counts=True)
    del labels
    reliability = counts / (counts + shrinkage_n0)
    pooled_excess = reliability * excess + (
        1.0 - reliability
    ) * np.mean(excess)
    return (
        float(np.mean(excess) / scale),
        float(np.mean(np.sort(pooled_excess)[-count:]) / scale),
    )


def _loss_coefficients(y, groups, baseline, delta):
    labels = np.unique(groups)
    error = np.asarray(baseline) - np.asarray(y)
    return np.asarray([
        (
            np.mean(error[groups == label] ** 2),
            np.mean(error[groups == label] * delta[groups == label]),
            np.mean(delta[groups == label] ** 2),
        )
        for label in labels
    ])


def _largest_feasible_alpha(
    y, groups, baseline, prior, modulation, epsilon, cvar_fraction,
    alpha_cap=1.0, shrinkage_n0=0.0,
):
    delta = modulation * (prior - baseline)
    coefficients = _loss_coefficients(y, groups, baseline, delta)
    alpha = np.linspace(0.0, alpha_cap, 101)
    losses = (
        coefficients[:, 0, None]
        + 2 * coefficients[:, 1, None] * alpha
        + coefficients[:, 2, None] * alpha**2
    )
    excess = losses - coefficients[:, 0, None]
    scale = max(float(np.mean(coefficients[:, 0])), 1e-12)
    count = max(1, int(np.ceil(cvar_fraction * len(coefficients))))
    mean_ratio = np.mean(excess, axis=0) / scale
    _, counts = np.unique(groups, return_counts=True)
    reliability = counts / (counts + shrinkage_n0)
    pooled_excess = (
        reliability[:, None] * excess
        + (1.0 - reliability[:, None]) * np.mean(excess, axis=0)
    )
    tail_ratio = (
        np.mean(np.sort(pooled_excess, axis=0)[-count:], axis=0) / scale
    )
    feasible = np.flatnonzero(
        (mean_ratio <= epsilon + 1e-12)
        & (tail_ratio <= epsilon + 1e-12)
    )
    index = int(feasible[-1])
    return float(alpha[index]), float(mean_ratio[index]), float(tail_ratio[index])


def fit_monotone_budget(
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prior: np.ndarray,
    distance: np.ndarray,
    disagreement: np.ndarray,
    *,
    decay_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    use_distance=True,
    use_disagreement=True,
    epsilon=0.02,
    cvar_fraction=0.20,
    alpha_cap=1.0,
    shrinkage_n0=0.0,
) -> MonotoneBudgetFit:
    """Select monotone decay and maximal feasible alpha without test labels."""
    distance_grid = decay_grid if use_distance else (0.0,)
    disagreement_grid = decay_grid if use_disagreement else (0.0,)
    candidates = []
    for distance_decay in distance_grid:
        for disagreement_decay in disagreement_grid:
            modulation = monotone_modulation(
                distance, disagreement, distance_decay, disagreement_decay
            )
            alpha, mean_ratio, tail_ratio = _largest_feasible_alpha(
                y, groups, baseline, prior, modulation, epsilon,
                cvar_fraction, alpha_cap, shrinkage_n0,
            )
            prediction = apply_prior_residual(
                baseline, prior, alpha, modulation
            )
            loss = float(np.mean(group_mse(y, prediction, groups)))
            candidates.append(MonotoneBudgetFit(
                alpha=alpha,
                distance_decay=float(distance_decay),
                disagreement_decay=float(disagreement_decay),
                validation_loss=loss,
                mean_excess_ratio=mean_ratio,
                tail_excess_ratio=tail_ratio,
            ))
    return min(candidates, key=lambda item: (
        item.validation_loss,
        item.distance_decay + item.disagreement_decay,
        item.alpha,
    ))


def crossfit_monotone_budget(
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prior: np.ndarray,
    distance: np.ndarray,
    disagreement: np.ndarray,
    **settings,
) -> CrossfitMonotoneResult:
    """Fit each held-out physical unit once and median-aggregate parameters."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    baseline = np.asarray(baseline, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=np.float64)
    if any(array.shape != y.shape for array in (
        groups, baseline, prior, distance, disagreement
    )):
        raise ValueError("all inputs must be aligned vectors")
    labels = np.unique(groups)
    if len(labels) < 5:
        raise ValueError("at least five physical units are required")
    oof = np.empty_like(y)
    parameters = []
    for held_out in labels:
        keep = groups != held_out
        fit = fit_monotone_budget(
            y[keep], groups[keep], baseline[keep], prior[keep],
            distance[keep], disagreement[keep], **settings,
        )
        parameters.append((
            fit.alpha, fit.distance_decay, fit.disagreement_decay
        ))
        take = ~keep
        modulation = monotone_modulation(
            distance[take], disagreement[take],
            fit.distance_decay, fit.disagreement_decay,
        )
        oof[take] = apply_prior_residual(
            baseline[take], prior[take], fit.alpha, modulation
        )
    epsilon = float(settings.get("epsilon", 0.02))
    cvar_fraction = float(settings.get("cvar_fraction", 0.20))
    shrinkage_n0 = float(settings.get("shrinkage_n0", 0.0))
    oof_mean, oof_tail = hierarchical_risk_ratios(
        y, groups, baseline, oof, cvar_fraction, shrinkage_n0
    )
    median = np.median(np.asarray(parameters), axis=0)
    modulation = monotone_modulation(
        distance, disagreement, median[1], median[2]
    )
    alpha, mean_ratio, tail_ratio = _largest_feasible_alpha(
        y, groups, baseline, prior, modulation, epsilon, cvar_fraction,
        alpha_cap=float(median[0]), shrinkage_n0=shrinkage_n0,
    )
    prediction = apply_prior_residual(baseline, prior, alpha, modulation)
    final = MonotoneBudgetFit(
        alpha=alpha,
        distance_decay=float(median[1]),
        disagreement_decay=float(median[2]),
        validation_loss=float(np.mean(group_mse(y, prediction, groups))),
        mean_excess_ratio=mean_ratio,
        tail_excess_ratio=tail_ratio,
    )
    return CrossfitMonotoneResult(
        final=final,
        fold_parameters=tuple(parameters),
        oof_prediction=oof,
        oof_mean_excess_ratio=oof_mean,
        oof_tail_excess_ratio=oof_tail,
        oof_feasible=oof_mean <= epsilon and oof_tail <= epsilon,
    )
