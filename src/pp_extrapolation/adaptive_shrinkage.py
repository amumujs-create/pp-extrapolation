"""Validation-only adaptive shrinkage between a matched NN and a PP prior."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveShrinkageDecision:
    candidate_index: int
    candidate_trust: float
    proposed_prior_weight: float
    prior_weight: float
    effective_trust: float
    evidence_level: str
    relative_rmse_gain: float
    unit_win_fraction: float
    worst_unit_rmse_ratio: float
    bootstrap_ci: tuple[float, float]
    bootstrap_probability_positive: float
    reason: str


def apply_adaptive_shrinkage(
    fallback_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    """Blend predictions; weight zero is the exact matched fallback."""
    fallback = np.asarray(fallback_prediction, dtype=np.float64)
    candidate = np.asarray(candidate_prediction, dtype=np.float64)
    if fallback.shape != candidate.shape:
        raise ValueError("fallback and candidate predictions must align")
    if not np.isfinite(prior_weight) or not 0 <= prior_weight <= 1:
        raise ValueError("prior_weight must be in [0, 1]")
    return fallback + float(prior_weight) * (candidate - fallback)


def select_adaptive_prior_shrinkage(
    y: np.ndarray,
    groups: np.ndarray,
    fallback_prediction: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_trusts: np.ndarray,
    *,
    strong_relative_gain: float = 0.02,
    strong_unit_win_fraction: float = 0.60,
    maximum_worst_unit_ratio: float = 1.10,
    weak_unit_win_fraction: float = 0.50,
    maximum_weak_prior_weight: float = 0.50,
    confidence: float = 0.95,
    bootstrap_replicates: int = 5000,
    seed: int = 20260910,
) -> AdaptiveShrinkageDecision:
    """Choose full, shrunk, or zero prior influence using validation only.

    Strong replicated evidence receives the complete validation-selected
    candidate. Directionally positive but uncertain evidence receives at most
    ``maximum_weak_prior_weight``. Non-positive pooled gain or no majority of
    physical units receives exact weight zero.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback_prediction, dtype=np.float64)
    candidates = np.asarray(candidate_predictions, dtype=np.float64)
    trusts = np.asarray(candidate_trusts, dtype=np.float64)
    if y.ndim != 1 or fallback.shape != y.shape:
        raise ValueError("y and fallback_prediction must be matching vectors")
    if candidates.ndim != 2 or candidates.shape[1] != len(y):
        raise ValueError("candidate_predictions must have shape [candidate, row]")
    if trusts.shape != (len(candidates),) or np.any(trusts <= 0):
        raise ValueError("candidate_trusts must be one positive value per candidate")
    labels = np.unique(groups)
    if len(labels) < 3:
        raise ValueError("at least three physical validation units are required")
    if not 0 < confidence < 1 or bootstrap_replicates < 100:
        raise ValueError("invalid bootstrap settings")
    if not 0 <= weak_unit_win_fraction <= strong_unit_win_fraction <= 1:
        raise ValueError("invalid unit-win thresholds")
    if not 0 <= maximum_weak_prior_weight <= 1:
        raise ValueError("invalid weak prior cap")

    fallback_rmse = float(np.sqrt(np.mean((fallback - y) ** 2)))
    candidate_rmse = np.sqrt(np.mean((candidates - y[None, :]) ** 2, axis=1))
    best = int(np.argmin(candidate_rmse))
    chosen = candidates[best]
    gain = float(
        (fallback_rmse - candidate_rmse[best]) / max(fallback_rmse, 1e-12)
    )
    positions = {label: np.flatnonzero(groups == label) for label in labels}
    ratios = []
    for label in labels:
        index = positions[label]
        base = np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
        prior = np.sqrt(np.mean((chosen[index] - y[index]) ** 2))
        ratios.append(float(prior / max(base, 1e-12)))
    ratios = np.asarray(ratios)
    win_fraction = float(np.mean(ratios < 1.0))
    worst_ratio = float(np.max(ratios))

    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap_replicates)
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(labels, len(labels), replace=True)
        index = np.concatenate([positions[label] for label in sampled])
        base = np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
        prior = np.sqrt(np.mean((chosen[index] - y[index]) ** 2))
        differences[replicate] = base - prior
    alpha = (1 - confidence) / 2
    low, high = np.quantile(differences, (alpha, 1 - alpha))
    interval = (float(low), float(high))
    probability = float(np.mean(differences > 0))

    strong = (
        gain >= strong_relative_gain
        and win_fraction >= strong_unit_win_fraction
        and worst_ratio <= maximum_worst_unit_ratio
        and low > 0
    )
    if strong:
        weight, level = 1.0, "strong"
        reason = "replicated validation evidence supports the full prior candidate"
    elif gain > 0 and win_fraction > weak_unit_win_fraction:
        gain_strength = min(gain / strong_relative_gain, 1.0)
        unit_strength = min(
            (win_fraction - weak_unit_win_fraction)
            / max(strong_unit_win_fraction - weak_unit_win_fraction, 1e-12),
            1.0,
        )
        proposed_weight = float(maximum_weak_prior_weight * np.sqrt(
            gain_strength * unit_strength
        ))
        # Keep the actual blended route, rather than the full candidate, within
        # the declared worst-unit validation envelope.
        weight = 0.0
        for trial in np.linspace(0.0, proposed_weight, 1001):
            blended = fallback + trial * (chosen - fallback)
            blended_ratios = []
            for label in labels:
                index = positions[label]
                base = np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
                risk = np.sqrt(np.mean((blended[index] - y[index]) ** 2))
                blended_ratios.append(risk / max(base, 1e-12))
            if max(blended_ratios) <= maximum_worst_unit_ratio:
                weight = float(trial)
        level = "weak"
        reason = (
            "directional evidence retains a continuously shrunk prior inside "
            "the worst-unit validation envelope"
        )
    else:
        proposed_weight, weight, level = 0.0, 0.0, "none"
        reason = "no positive pooled-and-unit validation direction; exact fallback"
    if strong:
        proposed_weight = weight
    trust = float(trusts[best])
    return AdaptiveShrinkageDecision(
        candidate_index=best,
        candidate_trust=trust,
        proposed_prior_weight=proposed_weight,
        prior_weight=weight,
        effective_trust=weight * trust,
        evidence_level=level,
        relative_rmse_gain=gain,
        unit_win_fraction=win_fraction,
        worst_unit_rmse_ratio=worst_ratio,
        bootstrap_ci=interval,
        bootstrap_probability_positive=probability,
        reason=reason,
    )
