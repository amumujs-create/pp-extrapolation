"""Train-only regime discovery with weak, safety-certified prior routes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk_budgeted_prior import apply_prior_residual
from .stability_first import (
    StabilityFirstDecision,
    regret_summary,
    select_stability_first_prior,
    unit_regret,
)


@dataclass(frozen=True)
class RegimeMap:
    center: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray
    temperature: float


@dataclass(frozen=True)
class AutoRegimePriorDecision:
    regime_map: RegimeMap
    regime_decisions: tuple[StabilityFirstDecision, ...]
    common_scale: float
    accepted_regimes: int
    validation_mean_regret: float
    validation_cvar_regret: float
    validation_max_regret: float


def _standardize(x, center, scale):
    return (np.asarray(x, dtype=np.float64) - center) / scale


def fit_regime_map(
    train_x: np.ndarray,
    *,
    n_regimes: int = 3,
    max_iter: int = 100,
) -> RegimeMap:
    """Fit deterministic robust-scaled k-means using train covariates only."""
    train_x = np.asarray(train_x, dtype=np.float64)
    if train_x.ndim != 2 or len(train_x) < n_regimes or n_regimes < 1:
        raise ValueError("train_x must support the requested regimes")
    center = np.median(train_x, axis=0)
    scale = np.quantile(train_x, 0.75, axis=0) - np.quantile(
        train_x, 0.25, axis=0
    )
    scale = np.where(scale > 1e-12, scale, np.std(train_x, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = _standardize(train_x, center, scale)

    centroid_indices = [int(np.argmin(np.sum(z**2, axis=1)))]
    for _ in range(1, n_regimes):
        distance = np.min(np.stack([
            np.sum((z - z[index]) ** 2, axis=1)
            for index in centroid_indices
        ]), axis=0)
        centroid_indices.append(int(np.argmax(distance)))
    centroids = z[centroid_indices].copy()

    for _ in range(max_iter):
        distance = np.stack([
            np.sum((z - centroid) ** 2, axis=1) for centroid in centroids
        ], axis=1)
        labels = np.argmin(distance, axis=1)
        updated = centroids.copy()
        for regime in range(n_regimes):
            take = labels == regime
            if np.any(take):
                updated[regime] = np.mean(z[take], axis=0)
        if np.allclose(updated, centroids, atol=1e-8, rtol=0):
            centroids = updated
            break
        centroids = updated
    final_distance = np.stack([
        np.sum((z - centroid) ** 2, axis=1) for centroid in centroids
    ], axis=1)
    nearest = np.min(final_distance, axis=1)
    positive = nearest[nearest > 1e-12]
    temperature = float(np.median(positive)) if len(positive) else 1.0
    return RegimeMap(center, scale, centroids, max(temperature, 1e-12))


def regime_probabilities(model: RegimeMap, x: np.ndarray) -> np.ndarray:
    """Return train-frozen soft regime membership for each row."""
    z = _standardize(x, model.center, model.scale)
    distance = np.stack([
        np.sum((z - centroid) ** 2, axis=1)
        for centroid in model.centroids
    ], axis=1)
    logits = -distance / model.temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    weight = np.exp(logits)
    return weight / np.sum(weight, axis=1, keepdims=True)


def _blend_routes(
    baseline: np.ndarray,
    priors: np.ndarray,
    trusts: np.ndarray,
    probabilities: np.ndarray,
    decisions: tuple[StabilityFirstDecision, ...],
) -> np.ndarray:
    routes = []
    for decision in decisions:
        if not decision.accepted:
            routes.append(np.asarray(baseline, dtype=np.float64))
            continue
        index = int(np.flatnonzero(np.isclose(
            trusts, decision.selected_trust
        ))[0])
        routes.append(apply_prior_residual(
            baseline, priors[index], decision.alpha
        ))
    return np.sum(probabilities.T * np.asarray(routes), axis=0)


def fit_auto_regime_prior(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prior_predictions: np.ndarray,
    trusts: np.ndarray,
    *,
    n_regimes=3,
    min_regime_groups=5,
    alpha_grid=np.linspace(0.0, 0.25, 6),
    mean_cap=0.02,
    cvar_cap=0.05,
    max_cap=0.10,
) -> AutoRegimePriorDecision:
    """Fit regime-local weak priors and a global raw-risk safety scale."""
    regime_map = fit_regime_map(train_x, n_regimes=n_regimes)
    probabilities = regime_probabilities(regime_map, validation_x)
    hard = np.argmax(probabilities, axis=1)
    decisions = []
    for regime in range(n_regimes):
        take = hard == regime
        if (
            np.sum(take) == 0
            or len(np.unique(groups[take])) < min_regime_groups
        ):
            decisions.append(StabilityFirstDecision(
                False, 0.0, 0.0,
                f"fewer than {min_regime_groups} regime validation units",
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (),
            ))
            continue
        decisions.append(select_stability_first_prior(
            y[take], groups[take], baseline[take],
            prior_predictions[:, take], trusts, alpha_grid=alpha_grid,
            mean_cap=mean_cap, cvar_cap=cvar_cap, max_cap=max_cap,
            min_groups=min_regime_groups,
        ))
    decision_tuple = tuple(decisions)
    blended = _blend_routes(
        baseline, prior_predictions, trusts, probabilities, decision_tuple
    )
    accepted = sum(decision.accepted for decision in decision_tuple)
    common_scale = 0.0
    final_summary = (0.0, 0.0, 0.0)
    if accepted:
        for scale in np.linspace(0.0, 1.0, 101):
            prediction = baseline + scale * (blended - baseline)
            summary = regret_summary(unit_regret(
                y, groups, baseline, prediction
            ))
            if (
                summary[0] <= mean_cap
                and summary[1] <= cvar_cap
                and summary[2] <= max_cap
            ):
                common_scale = float(scale)
                final_summary = summary
    return AutoRegimePriorDecision(
        regime_map=regime_map,
        regime_decisions=decision_tuple,
        common_scale=common_scale,
        accepted_regimes=accepted,
        validation_mean_regret=final_summary[0],
        validation_cvar_regret=final_summary[1],
        validation_max_regret=final_summary[2],
    )


def predict_auto_regime_prior(
    decision: AutoRegimePriorDecision,
    x: np.ndarray,
    baseline: np.ndarray,
    prior_predictions: np.ndarray,
    trusts: np.ndarray,
    *,
    hard: bool = False,
) -> np.ndarray:
    """Apply frozen soft regime routes without test-batch fitting."""
    probabilities = regime_probabilities(decision.regime_map, x)
    if hard:
        labels = np.argmax(probabilities, axis=1)
        probabilities = np.eye(probabilities.shape[1])[labels]
    blended = _blend_routes(
        baseline, prior_predictions, np.asarray(trusts), probabilities,
        decision.regime_decisions,
    )
    return baseline + decision.common_scale * (blended - baseline)
