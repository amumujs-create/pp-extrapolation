"""Cross-fitted consensus over validation-approved prior continuations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .residual_authority import distance_shell_edges
from .risk_budgeted_prior import group_mse


@dataclass(frozen=True)
class PriorSetPolicy:
    edges: tuple[float, ...]
    approved_experts: tuple[tuple[int, ...], ...]
    disagreement_limits: tuple[float, ...]
    nested: bool
    validation_loss: float
    relative_gain_vs_fallback: float
    active_fraction: float


@dataclass(frozen=True)
class CrossfitPriorSetResult:
    policy: PriorSetPolicy
    oof_prediction: np.ndarray
    oof_relative_gain_vs_fallback: float
    oof_mean_harm_ratio: float
    oof_tail_harm_ratio: float
    approved: bool
    fold_sets: tuple[tuple[tuple[int, ...], ...], ...]


def _shell_index(distance: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges[1:-1]), distance, side="right")


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


def apply_prior_set_consensus(
    experts: np.ndarray,
    fallback: np.ndarray,
    distance: np.ndarray,
    policy: PriorSetPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prediction, active mask, and prior-envelope width."""
    experts = np.asarray(experts, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if (
        experts.ndim != 2
        or experts.shape[1] != len(fallback)
        or distance.shape != fallback.shape
    ):
        raise ValueError("experts, fallback, and distance do not align")
    shell = _shell_index(distance, policy.edges)
    prediction = fallback.copy()
    active = np.zeros(len(fallback), dtype=bool)
    width = np.full(len(fallback), np.inf)
    for index, approved in enumerate(policy.approved_experts):
        if not approved:
            continue
        mask = shell == index
        values = experts[np.asarray(approved), :][:, mask]
        if values.size == 0:
            continue
        local_width = np.max(values, axis=0) - np.min(values, axis=0)
        accepted = local_width <= policy.disagreement_limits[index] + 1e-12
        rows = np.flatnonzero(mask)
        chosen = rows[accepted]
        prediction[chosen] = np.median(values[:, accepted], axis=0)
        active[chosen] = True
        width[rows] = local_width
    return prediction, active, width


def fit_prior_set_policy(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    experts: np.ndarray,
    distance: np.ndarray,
    *,
    edges: tuple[float, ...] | None = None,
    epsilon: float = 0.02,
    cvar_fraction: float = 0.20,
    minimum_groups_per_shell: int = 2,
    disagreement_quantile: float = 0.90,
    nested: bool = True,
) -> PriorSetPolicy:
    """Approve prior continuations by shell and freeze their envelope limits."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback, dtype=np.float64)
    experts = np.asarray(experts, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if (
        y.ndim != 1
        or fallback.shape != y.shape
        or distance.shape != y.shape
        or experts.ndim != 2
        or experts.shape[1] != len(y)
    ):
        raise ValueError("inputs do not align")
    if not np.isfinite(np.concatenate([
        y, fallback, experts.ravel(), distance
    ])).all() or np.any(distance < 0):
        raise ValueError("numeric inputs must be finite and distances nonnegative")
    if (
        epsilon < 0
        or not 0 < cvar_fraction <= 1
        or minimum_groups_per_shell < 1
        or not 0 < disagreement_quantile <= 1
    ):
        raise ValueError("invalid prior-set settings")
    shell_edges = edges or distance_shell_edges(distance)
    shell = _shell_index(distance, shell_edges)
    sets = []
    limits = []
    previous = set(range(len(experts)))
    for index in range(len(shell_edges) - 1):
        mask = shell == index
        labels = np.unique(groups[mask])
        approved = set()
        if len(labels) >= minimum_groups_per_shell:
            for expert in range(len(experts)):
                mean_harm, tail_harm = _harm_ratios(
                    y[mask],
                    groups[mask],
                    fallback[mask],
                    experts[expert, mask],
                    cvar_fraction,
                )
                if mean_harm < -1e-12 and tail_harm <= epsilon + 1e-12:
                    approved.add(expert)
        if nested:
            approved &= previous
            previous = approved
        selected = tuple(sorted(approved))
        sets.append(selected)
        if selected and np.any(mask):
            values = experts[np.asarray(selected), :][:, mask]
            width = np.max(values, axis=0) - np.min(values, axis=0)
            limits.append(float(np.quantile(width, disagreement_quantile)))
        else:
            limits.append(0.0)
    provisional = PriorSetPolicy(
        edges=tuple(float(value) for value in shell_edges),
        approved_experts=tuple(sets),
        disagreement_limits=tuple(limits),
        nested=bool(nested),
        validation_loss=0.0,
        relative_gain_vs_fallback=0.0,
        active_fraction=0.0,
    )
    prediction, active, _ = apply_prior_set_consensus(
        experts, fallback, distance, provisional
    )
    baseline_loss = float(np.mean(group_mse(y, fallback, groups)))
    loss = float(np.mean(group_mse(y, prediction, groups)))
    return PriorSetPolicy(
        edges=provisional.edges,
        approved_experts=provisional.approved_experts,
        disagreement_limits=provisional.disagreement_limits,
        nested=provisional.nested,
        validation_loss=loss,
        relative_gain_vs_fallback=(baseline_loss - loss) / max(
            baseline_loss, 1e-12
        ),
        active_fraction=float(np.mean(active)),
    )


def crossfit_prior_set_consensus(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    experts: np.ndarray,
    distance: np.ndarray,
    *,
    minimum_relative_gain: float = 0.0,
    **settings,
) -> CrossfitPriorSetResult:
    """Leave one physical unit out when fitting every prior-set policy."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback, dtype=np.float64)
    experts = np.asarray(experts, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    labels = np.unique(groups)
    if len(labels) < 5:
        raise ValueError("at least five physical validation units are required")
    edges = settings.pop("edges", None) or distance_shell_edges(distance)
    oof = np.empty_like(y)
    fold_sets = []
    for held_out in labels:
        keep = groups != held_out
        policy = fit_prior_set_policy(
            y[keep],
            groups[keep],
            fallback[keep],
            experts[:, keep],
            distance[keep],
            edges=edges,
            **settings,
        )
        take = ~keep
        oof[take] = apply_prior_set_consensus(
            experts[:, take], fallback[take], distance[take], policy
        )[0]
        fold_sets.append(policy.approved_experts)
    baseline_loss = float(np.mean(group_mse(y, fallback, groups)))
    oof_loss = float(np.mean(group_mse(y, oof, groups)))
    gain = (baseline_loss - oof_loss) / max(baseline_loss, 1e-12)
    epsilon = float(settings.get("epsilon", 0.02))
    cvar_fraction = float(settings.get("cvar_fraction", 0.20))
    mean_harm, tail_harm = _harm_ratios(
        y, groups, fallback, oof, cvar_fraction
    )
    approved = (
        gain > minimum_relative_gain + 1e-12
        and mean_harm <= epsilon
        and tail_harm <= epsilon
    )
    policy = fit_prior_set_policy(
        y, groups, fallback, experts, distance, edges=edges, **settings
    )
    if not approved:
        policy = PriorSetPolicy(
            edges=policy.edges,
            approved_experts=tuple(() for _ in policy.approved_experts),
            disagreement_limits=tuple(0.0 for _ in policy.disagreement_limits),
            nested=policy.nested,
            validation_loss=baseline_loss,
            relative_gain_vs_fallback=0.0,
            active_fraction=0.0,
        )
    return CrossfitPriorSetResult(
        policy=policy,
        oof_prediction=oof,
        oof_relative_gain_vs_fallback=gain,
        oof_mean_harm_ratio=mean_harm,
        oof_tail_harm_ratio=tail_harm,
        approved=approved,
        fold_sets=tuple(fold_sets),
    )
