"""Train-only heterogeneous priors with prior-specific approval sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .residual_authority import distance_shell_edges
from .risk_budgeted_prior import group_mse


@dataclass(frozen=True)
class AffinePriorFit:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class MonotoneHealthFit:
    interpolator: IsotonicRegression
    health_min: float
    tail_slope: float
    y_at_min: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class HistoryRateFit:
    intercept: float
    coefficient: float
    rate_floor: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class StructuralPriorSetPolicy:
    names: tuple[str, ...]
    edges: tuple[tuple[float, ...], ...]
    approved_shells: tuple[tuple[int, ...], ...]
    disagreement_limits: tuple[float, ...]
    ceilings: tuple[float, ...]
    minimum_set_size: int
    nested: bool
    validation_loss: float
    relative_gain_vs_fallback: float
    active_fraction: float


@dataclass(frozen=True)
class CrossfitStructuralPriorResult:
    policy: StructuralPriorSetPolicy
    oof_prediction: np.ndarray
    oof_relative_gain_vs_fallback: float
    oof_mean_harm_ratio: float
    oof_tail_harm_ratio: float
    approved: bool
    fold_sets: tuple[tuple[tuple[int, ...], ...], ...]


def _clip(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float64), lower, upper)


def _harm_ratios(y, groups, fallback, prediction, cvar_fraction):
    baseline = group_mse(y, fallback, groups)
    candidate = group_mse(y, prediction, groups)
    excess = candidate - baseline
    scale = max(float(np.mean(baseline)), 1e-12)
    count = max(1, int(np.ceil(cvar_fraction * len(excess))))
    return (
        float(np.mean(excess) / scale),
        float(np.mean(np.sort(excess)[-count:]) / scale),
    )


def _weighted_ridge(x, y, groups, alpha):
    labels = np.unique(groups)
    weights = np.zeros(len(y), dtype=np.float64)
    for label in labels:
        mask = groups == label
        weights[mask] = 1.0 / max(int(np.sum(mask)), 1)
    weights /= weights.sum()
    total = float(np.sum(weights))
    x_mean = np.sum(weights[:, None] * x, axis=0) / total
    y_mean = float(np.sum(weights * y) / total)
    centered_x = x - x_mean
    coefficient = np.linalg.solve(
        centered_x.T @ (weights[:, None] * centered_x)
        + float(alpha) * np.eye(x.shape[1]),
        centered_x.T @ (weights * (y - y_mean)),
    )
    return coefficient, y_mean - float(x_mean @ coefficient)


def fit_affine_prior(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
) -> AffinePriorFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape:
        raise ValueError("affine prior inputs do not align")
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - center) / scale
    candidates = []
    for alpha in alphas:
        coefficient, intercept = _weighted_ridge(z, y, groups, alpha)
        prediction = _clip(z @ coefficient + intercept, float(np.min(y)), float(np.max(y)))
        loss = float(np.mean(group_mse(y, prediction, groups)))
        candidates.append((loss, alpha, coefficient, intercept))
    _, _, coefficient, intercept = min(candidates, key=lambda row: (row[0], row[1]))
    return AffinePriorFit(
        center=center,
        scale=scale,
        coefficient=coefficient,
        intercept=float(intercept),
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
    )


def predict_affine_prior(model: AffinePriorFit, x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - model.center) / model.scale
    return _clip(z @ model.coefficient + model.intercept, model.y_min, model.y_max)


def fit_monotone_health_prior(
    health: np.ndarray,
    y: np.ndarray,
    *,
    tail_fraction: float = 0.20,
) -> MonotoneHealthFit:
    health = np.asarray(health, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if health.shape != y.shape or len(y) < 4:
        raise ValueError("monotone prior needs aligned nonempty vectors")
    interpolator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    interpolator.fit(health, y)
    order = np.argsort(health)
    count = max(2, int(np.ceil(tail_fraction * len(health))))
    low = order[:count]
    design = np.column_stack([np.ones(count), health[low] - health[low].max()])
    slope = float(np.linalg.lstsq(design, y[low], rcond=None)[0][1])
    health_min = float(np.min(health))
    return MonotoneHealthFit(
        interpolator=interpolator,
        health_min=health_min,
        tail_slope=max(slope, 0.0),
        y_at_min=float(interpolator.predict(np.asarray([health_min]))[0]),
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
    )


def predict_monotone_health_prior(
    model: MonotoneHealthFit, health: np.ndarray
) -> np.ndarray:
    health = np.asarray(health, dtype=np.float64)
    inside = model.interpolator.predict(health)
    below = health < model.health_min
    tail = model.y_at_min + model.tail_slope * (health - model.health_min)
    return _clip(np.where(below, tail, inside), model.y_min, model.y_max)


def _eta(health: np.ndarray, rate: np.ndarray, rate_floor: float) -> np.ndarray:
    return np.asarray(health, dtype=np.float64) / np.maximum(
        -np.asarray(rate, dtype=np.float64), rate_floor
    )


def fit_history_rate_prior(
    health: np.ndarray,
    rate: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> HistoryRateFit:
    health = np.asarray(health, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if not (health.shape == rate.shape == y.shape == groups.shape):
        raise ValueError("history-rate prior inputs do not align")
    negative = -rate[rate < 0]
    rate_floor = float(np.quantile(negative, 0.10)) if len(negative) else 1e-6
    rate_floor = max(rate_floor, 1e-6)
    eta = _eta(health, rate, rate_floor)
    design = np.column_stack([np.ones(len(y)), eta])
    weights, intercept = _weighted_ridge(design, y, groups, 1.0)
    coefficient = float(weights[1])
    return HistoryRateFit(
        intercept=float(intercept),
        coefficient=float(coefficient),
        rate_floor=rate_floor,
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
    )


def predict_history_rate_prior(
    model: HistoryRateFit, health: np.ndarray, rate: np.ndarray
) -> np.ndarray:
    eta = _eta(health, rate, model.rate_floor)
    return _clip(model.intercept + model.coefficient * eta, model.y_min, model.y_max)


def health_distance(health: np.ndarray, train_min: float, scale: float) -> np.ndarray:
    return np.maximum(train_min - np.asarray(health, dtype=np.float64), 0.0) / max(
        float(scale), 1e-12
    )


def _shell_index(distance: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges[1:-1]), distance, side="right")


def apply_structural_prior_set(
    experts: np.ndarray,
    fallback: np.ndarray,
    distances: np.ndarray,
    policy: StructuralPriorSetPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prediction, active mask, and retained-set width."""
    experts = np.asarray(experts, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if (
        experts.ndim != 2
        or distances.shape != experts.shape
        or fallback.shape != (experts.shape[1],)
    ):
        raise ValueError("experts, distances, and fallback do not align")
    if len(policy.approved_shells) != len(experts):
        raise ValueError("policy does not match the number of priors")
    prediction = fallback.copy()
    active = np.zeros(len(fallback), dtype=bool)
    width = np.full(len(fallback), np.inf)
    for row in range(len(fallback)):
        chosen = []
        values = []
        for index, approved in enumerate(policy.approved_shells):
            distance = distances[index, row]
            if distance > policy.ceilings[index] + 1e-12:
                continue
            shell = int(_shell_index(np.asarray([distance]), policy.edges[index])[0])
            if shell in approved:
                chosen.append(index)
                values.append(experts[index, row])
        if len(chosen) < policy.minimum_set_size or not values:
            continue
        local_width = float(np.max(values) - np.min(values))
        width[row] = local_width
        limit = policy.disagreement_limits[min(len(values), 2) - 1]
        if local_width > limit + 1e-12:
            continue
        prediction[row] = float(np.median(values))
        active[row] = True
    return prediction, active, width


def fit_structural_prior_set_policy(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    experts: np.ndarray,
    distances: np.ndarray,
    *,
    names: tuple[str, ...] | None = None,
    edges: tuple[tuple[float, ...], ...] | None = None,
    ceilings: tuple[float, ...] | None = None,
    ceiling_quantile: float = 0.99,
    epsilon: float = 0.02,
    cvar_fraction: float = 0.20,
    minimum_groups_per_shell: int = 2,
    disagreement_quantile: float = 0.90,
    minimum_set_size: int = 1,
    nested: bool = True,
) -> StructuralPriorSetPolicy:
    """Approve each structural prior in its own distance shells."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback, dtype=np.float64)
    experts = np.asarray(experts, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if (
        y.ndim != 1
        or fallback.shape != y.shape
        or experts.ndim != 2
        or experts.shape[1] != len(y)
        or distances.shape != experts.shape
    ):
        raise ValueError("inputs do not align")
    if not np.isfinite(np.concatenate([
        y, fallback, experts.ravel(), distances.ravel()
    ])).all() or np.any(distances < 0):
        raise ValueError("numeric inputs must be finite and distances nonnegative")
    if (
        epsilon < 0
        or not 0 < cvar_fraction <= 1
        or minimum_groups_per_shell < 1
        or not 0 < disagreement_quantile <= 1
        or not 0 < ceiling_quantile <= 1
        or minimum_set_size < 1
    ):
        raise ValueError("invalid structural prior-set settings")
    count = len(experts)
    prior_names = names or tuple(f"prior_{index}" for index in range(count))
    if len(prior_names) != count:
        raise ValueError("names must match the number of priors")
    fitted_edges = []
    fitted_ceilings = []
    approved = []
    for index in range(count):
        local_edges = (
            edges[index] if edges is not None
            else distance_shell_edges(distances[index])
        )
        fitted_edges.append(tuple(float(value) for value in local_edges))
        ceiling = (
            ceilings[index] if ceilings is not None
            else float(np.quantile(distances[index], ceiling_quantile))
        )
        fitted_ceilings.append(float(ceiling))
        shell = _shell_index(distances[index], local_edges)
        kept: list[int] = []
        previous = True
        for shell_index in range(len(local_edges) - 1):
            mask = (
                (shell == shell_index)
                & (distances[index] <= ceiling + 1e-12)
            )
            labels = np.unique(groups[mask])
            accept = False
            if previous and len(labels) >= minimum_groups_per_shell:
                mean_harm, tail_harm = _harm_ratios(
                    y[mask],
                    groups[mask],
                    fallback[mask],
                    experts[index, mask],
                    cvar_fraction,
                )
                accept = mean_harm < -1e-12 and tail_harm <= epsilon + 1e-12
            if nested:
                previous = accept
            if accept:
                kept.append(shell_index)
        approved.append(tuple(kept))
    limits = [0.0, 0.0]
    for size, slot in ((1, 0), (2, 1)):
        widths = []
        for row in range(len(y)):
            values = [
                experts[index, row]
                for index, shells in enumerate(approved)
                if int(_shell_index(
                    np.asarray([distances[index, row]]), fitted_edges[index]
                )[0]) in shells
                and distances[index, row] <= fitted_ceilings[index] + 1e-12
            ]
            if len(values) >= size:
                widths.append(float(np.max(values) - np.min(values)))
        limits[slot] = (
            float(np.quantile(widths, disagreement_quantile)) if widths else 0.0
        )
    provisional = StructuralPriorSetPolicy(
        names=tuple(prior_names),
        edges=tuple(fitted_edges),
        approved_shells=tuple(approved),
        disagreement_limits=tuple(limits),
        ceilings=tuple(fitted_ceilings),
        minimum_set_size=int(minimum_set_size),
        nested=bool(nested),
        validation_loss=0.0,
        relative_gain_vs_fallback=0.0,
        active_fraction=0.0,
    )
    prediction, active, _ = apply_structural_prior_set(
        experts, fallback, distances, provisional
    )
    baseline_loss = float(np.mean(group_mse(y, fallback, groups)))
    loss = float(np.mean(group_mse(y, prediction, groups)))
    return StructuralPriorSetPolicy(
        names=provisional.names,
        edges=provisional.edges,
        approved_shells=provisional.approved_shells,
        disagreement_limits=provisional.disagreement_limits,
        ceilings=provisional.ceilings,
        minimum_set_size=provisional.minimum_set_size,
        nested=provisional.nested,
        validation_loss=loss,
        relative_gain_vs_fallback=(baseline_loss - loss) / max(baseline_loss, 1e-12),
        active_fraction=float(np.mean(active)),
    )


def crossfit_structural_prior_set(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    experts: np.ndarray,
    distances: np.ndarray,
    *,
    minimum_relative_gain: float = 0.0,
    **settings,
) -> CrossfitStructuralPriorResult:
    """Leave one physical unit out when fitting the approval policy."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback, dtype=np.float64)
    experts = np.asarray(experts, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.unique(groups)
    if len(labels) < 5:
        raise ValueError("at least five physical validation units are required")
    oof = np.empty_like(y)
    fold_sets = []
    shared = {
        "edges": settings.pop("edges", None),
        "ceilings": settings.pop("ceilings", None),
    }
    for held_out in labels:
        keep = groups != held_out
        policy = fit_structural_prior_set_policy(
            y[keep],
            groups[keep],
            fallback[keep],
            experts[:, keep],
            distances[:, keep],
            **{key: value for key, value in shared.items() if value is not None},
            **settings,
        )
        take = ~keep
        oof[take] = apply_structural_prior_set(
            experts[:, take], fallback[take], distances[:, take], policy
        )[0]
        fold_sets.append(policy.approved_shells)
    baseline_loss = float(np.mean(group_mse(y, fallback, groups)))
    oof_loss = float(np.mean(group_mse(y, oof, groups)))
    gain = (baseline_loss - oof_loss) / max(baseline_loss, 1e-12)
    epsilon = float(settings.get("epsilon", 0.02))
    cvar_fraction = float(settings.get("cvar_fraction", 0.20))
    mean_harm, tail_harm = _harm_ratios(y, groups, fallback, oof, cvar_fraction)
    approved = (
        gain > minimum_relative_gain + 1e-12
        and mean_harm <= epsilon
        and tail_harm <= epsilon
    )
    policy = fit_structural_prior_set_policy(
        y, groups, fallback, experts, distances,
        **{key: value for key, value in shared.items() if value is not None},
        **settings,
    )
    if not approved:
        policy = StructuralPriorSetPolicy(
            names=policy.names,
            edges=policy.edges,
            approved_shells=tuple(() for _ in policy.approved_shells),
            disagreement_limits=policy.disagreement_limits,
            ceilings=policy.ceilings,
            minimum_set_size=policy.minimum_set_size,
            nested=policy.nested,
            validation_loss=baseline_loss,
            relative_gain_vs_fallback=0.0,
            active_fraction=0.0,
        )
    return CrossfitStructuralPriorResult(
        policy=policy,
        oof_prediction=oof,
        oof_relative_gain_vs_fallback=gain,
        oof_mean_harm_ratio=mean_harm,
        oof_tail_harm_ratio=tail_harm,
        approved=approved,
        fold_sets=tuple(fold_sets),
    )
