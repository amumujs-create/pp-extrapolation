"""Cross-fitted, distance-indexed authority for a frozen prior residual.

The authority is not a model-selection gate.  It bounds how far an already
trained residual candidate may move a frozen prior inside each validation
distance shell.  Unsupported shells use the declared prior-off fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .risk_budgeted_prior import group_mse


@dataclass(frozen=True)
class ResidualAuthorityPolicy:
    edges: tuple[float, ...]
    authorities: tuple[float | None, ...]
    validation_loss: float
    relative_gain_vs_fallback: float
    mean_harm_ratio: float
    tail_harm_ratio: float
    active_shells: int


@dataclass(frozen=True)
class CrossfitAuthorityResult:
    policy: ResidualAuthorityPolicy
    oof_prediction: np.ndarray
    oof_relative_gain_vs_fallback: float
    oof_mean_harm_ratio: float
    oof_tail_harm_ratio: float
    approved: bool
    fold_authorities: tuple[tuple[float | None, ...], ...]


def distance_shell_edges(
    distance: np.ndarray,
    quantiles: tuple[float, ...] = (0.0, 0.50, 0.80, 1.0),
) -> tuple[float, ...]:
    """Create label-free validation-distance shells with an infinite last edge."""
    values = np.asarray(distance, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("distance must be a nonempty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("distance must be finite and nonnegative")
    if (
        len(quantiles) < 2
        or quantiles[0] != 0
        or quantiles[-1] != 1
        or any(right <= left for left, right in zip(quantiles, quantiles[1:]))
    ):
        raise ValueError("quantiles must increase strictly from zero to one")
    internal = np.quantile(values, quantiles[1:-1])
    finite = [0.0]
    for value in internal:
        value = float(value)
        if value > finite[-1] + 1e-12:
            finite.append(value)
    finite.append(float("inf"))
    return tuple(finite)


def _shell_index(distance: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges[1:-1]), distance, side="right")


def apply_residual_authority(
    prior: np.ndarray,
    candidate: np.ndarray,
    fallback: np.ndarray,
    distance: np.ndarray,
    policy: ResidualAuthorityPolicy,
) -> np.ndarray:
    """Apply the frozen authority policy; ``None`` means exact fallback."""
    prior = np.asarray(prior, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if any(value.shape != prior.shape for value in (candidate, fallback, distance)):
        raise ValueError("predictions and distances must be aligned vectors")
    shell = _shell_index(distance, policy.edges)
    prediction = fallback.copy()
    for index, authority in enumerate(policy.authorities):
        if authority is None:
            continue
        mask = shell == index
        prediction[mask] = prior[mask] + authority * (
            candidate[mask] - prior[mask]
        )
    return prediction


def _harm_ratios(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    prediction: np.ndarray,
    cvar_fraction: float,
) -> tuple[float, float]:
    baseline_loss = group_mse(y, fallback, groups)
    candidate_loss = group_mse(y, prediction, groups)
    excess = candidate_loss - baseline_loss
    scale = max(float(np.mean(baseline_loss)), 1e-12)
    count = max(1, int(np.ceil(cvar_fraction * len(excess))))
    return (
        float(np.mean(excess) / scale),
        float(np.mean(np.sort(excess)[-count:]) / scale),
    )


def _candidate_sequences(
    shell_count: int,
    authority_grid: tuple[float, ...],
    enforce_monotone: bool,
):
    # Once evidence fails at a distance shell, all farther shells also fall back.
    yield (None,) * shell_count
    for active_count in range(1, shell_count + 1):
        for values in product(authority_grid, repeat=active_count):
            if (
                not enforce_monotone
                or all(
                    left + 1e-12 >= right
                    for left, right in zip(values, values[1:])
                )
            ):
                yield tuple(float(value) for value in values) + (
                    (None,) * (shell_count - active_count)
                )


def fit_residual_authority(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    prior: np.ndarray,
    candidate: np.ndarray,
    distance: np.ndarray,
    *,
    edges: tuple[float, ...] | None = None,
    authority_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
    epsilon: float = 0.02,
    cvar_fraction: float = 0.20,
    minimum_groups_per_shell: int = 2,
    enforce_monotone: bool = True,
) -> ResidualAuthorityPolicy:
    """Fit the lowest-loss monotone authority satisfying shell harm budgets."""
    arrays = [
        np.asarray(value)
        for value in (y, groups, fallback, prior, candidate, distance)
    ]
    y, groups, fallback, prior, candidate, distance = arrays
    if y.ndim != 1 or any(value.shape != y.shape for value in arrays[1:]):
        raise ValueError("all inputs must be aligned vectors")
    if not np.isfinite(np.concatenate([
        y.astype(float), fallback.astype(float), prior.astype(float),
        candidate.astype(float), distance.astype(float),
    ])).all():
        raise ValueError("numeric inputs must be finite")
    if np.any(distance < 0):
        raise ValueError("distance must be nonnegative")
    if (
        epsilon < 0
        or not 0 < cvar_fraction <= 1
        or minimum_groups_per_shell < 1
        or len(authority_grid) < 2
        or any(value < 0 or value > 1 for value in authority_grid)
    ):
        raise ValueError("invalid authority settings")
    if any(right <= left for left, right in zip(authority_grid, authority_grid[1:])):
        raise ValueError("authority_grid must increase strictly")
    shell_edges = edges or distance_shell_edges(distance)
    if (
        len(shell_edges) < 2
        or shell_edges[0] != 0
        or not np.isinf(shell_edges[-1])
        or any(right <= left for left, right in zip(shell_edges, shell_edges[1:]))
    ):
        raise ValueError("edges must increase from zero to infinity")
    shell = _shell_index(distance, shell_edges)
    shell_count = len(shell_edges) - 1
    fallback_loss = float(np.mean(group_mse(y, fallback, groups)))
    feasible: list[ResidualAuthorityPolicy] = []
    for authorities in _candidate_sequences(
        shell_count, authority_grid, enforce_monotone
    ):
        prediction = fallback.copy()
        valid = True
        for index, authority in enumerate(authorities):
            if authority is None:
                continue
            mask = shell == index
            if len(np.unique(groups[mask])) < minimum_groups_per_shell:
                valid = False
                break
            shell_prediction = prior[mask] + authority * (
                candidate[mask] - prior[mask]
            )
            mean_harm, tail_harm = _harm_ratios(
                y[mask], groups[mask], fallback[mask], shell_prediction,
                cvar_fraction,
            )
            if mean_harm > epsilon + 1e-12 or tail_harm > epsilon + 1e-12:
                valid = False
                break
            prediction[mask] = shell_prediction
        if not valid:
            continue
        loss = float(np.mean(group_mse(y, prediction, groups)))
        mean_harm, tail_harm = _harm_ratios(
            y, groups, fallback, prediction, cvar_fraction
        )
        feasible.append(ResidualAuthorityPolicy(
            edges=tuple(float(value) for value in shell_edges),
            authorities=authorities,
            validation_loss=loss,
            relative_gain_vs_fallback=(fallback_loss - loss) / max(
                fallback_loss, 1e-12
            ),
            mean_harm_ratio=mean_harm,
            tail_harm_ratio=tail_harm,
            active_shells=sum(value is not None for value in authorities),
        ))
    return min(
        feasible,
        key=lambda item: (
            item.validation_loss,
            item.active_shells,
            sum(value or 0.0 for value in item.authorities),
        ),
    )


def crossfit_residual_authority(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    prior: np.ndarray,
    candidate: np.ndarray,
    distance: np.ndarray,
    *,
    minimum_relative_gain: float = 0.0,
    **settings,
) -> CrossfitAuthorityResult:
    """Leave one physical unit out when estimating every OOF authority."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if any(value.shape != y.shape for value in (
        groups, fallback, prior, candidate, distance
    )):
        raise ValueError("all inputs must be aligned vectors")
    labels = np.unique(groups)
    if len(labels) < 5:
        raise ValueError("at least five physical validation units are required")
    edges = settings.pop("edges", None) or distance_shell_edges(distance)
    oof = np.empty_like(y)
    fold_authorities = []
    for held_out in labels:
        keep = groups != held_out
        policy = fit_residual_authority(
            y[keep], groups[keep], fallback[keep], prior[keep],
            candidate[keep], distance[keep], edges=edges, **settings,
        )
        take = ~keep
        oof[take] = apply_residual_authority(
            prior[take], candidate[take], fallback[take], distance[take],
            policy,
        )
        fold_authorities.append(policy.authorities)
    fallback_loss = float(np.mean(group_mse(y, fallback, groups)))
    oof_loss = float(np.mean(group_mse(y, oof, groups)))
    gain = (fallback_loss - oof_loss) / max(fallback_loss, 1e-12)
    cvar_fraction = float(settings.get("cvar_fraction", 0.20))
    mean_harm, tail_harm = _harm_ratios(
        y, groups, fallback, oof, cvar_fraction
    )
    epsilon = float(settings.get("epsilon", 0.02))
    approved = (
        gain > minimum_relative_gain + 1e-12
        and mean_harm <= epsilon
        and tail_harm <= epsilon
    )
    policy = fit_residual_authority(
        y, groups, fallback, prior, candidate, distance,
        edges=edges, **settings,
    )
    if not approved:
        policy = ResidualAuthorityPolicy(
            edges=policy.edges,
            authorities=(None,) * len(policy.authorities),
            validation_loss=fallback_loss,
            relative_gain_vs_fallback=0.0,
            mean_harm_ratio=0.0,
            tail_harm_ratio=0.0,
            active_shells=0,
        )
    return CrossfitAuthorityResult(
        policy=policy,
        oof_prediction=oof,
        oof_relative_gain_vs_fallback=gain,
        oof_mean_harm_ratio=mean_harm,
        oof_tail_harm_ratio=tail_harm,
        approved=approved,
        fold_authorities=tuple(fold_authorities),
    )
