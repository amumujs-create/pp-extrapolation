"""Conservative validation-only selection for prior/NN continuations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriorTrustDecision:
    """Decision made without test labels or test-batch statistics."""

    accepted: bool
    candidate_index: int | None
    relative_rmse_gain: float
    bootstrap_ci: tuple[float, float]
    unit_win_fraction: float
    worst_unit_rmse_ratio: float
    reason: str


def select_prior_trust(
    y: np.ndarray,
    groups: np.ndarray,
    fallback_prediction: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    min_relative_gain: float = 0.02,
    min_unit_win_fraction: float = 0.60,
    max_worst_unit_rmse_ratio: float = 1.10,
    confidence: float = 0.95,
    bootstrap_replicates: int = 5000,
    seed: int = 20260910,
) -> PriorTrustDecision:
    """Approve one prior route only with replicated validation improvement.

    Candidates are compared with the exact matched-NN fallback.  The selected
    candidate must improve pooled RMSE by ``min_relative_gain`` and have a
    positive physical-unit bootstrap lower bound for ``fallback RMSE -
    candidate RMSE``.  Otherwise the caller must use trust zero.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback_prediction, dtype=np.float64)
    candidates = np.asarray(candidate_predictions, dtype=np.float64)
    if y.ndim != 1 or fallback.shape != y.shape:
        raise ValueError("y and fallback_prediction must be matching vectors")
    if candidates.ndim != 2 or candidates.shape[1] != len(y):
        raise ValueError("candidate_predictions must have shape [candidate, row]")
    if len(np.unique(groups)) < 3:
        return PriorTrustDecision(False, None, 0.0, (0.0, 0.0), 0.0, float("inf"),
                                  "fewer than three validation units")
    if not 0 < confidence < 1 or bootstrap_replicates < 100:
        raise ValueError("invalid bootstrap settings")

    fallback_rmse = float(np.sqrt(np.mean((fallback - y) ** 2)))
    candidate_rmse = np.sqrt(np.mean((candidates - y[None, :]) ** 2, axis=1))
    best_index = int(np.argmin(candidate_rmse))
    gain = (fallback_rmse - float(candidate_rmse[best_index])) / max(fallback_rmse, 1e-12)
    if gain < min_relative_gain:
        return PriorTrustDecision(False, None, gain, (0.0, 0.0), 0.0, float("inf"),
                                  "validation RMSE gain below required margin")

    labels = np.unique(groups)
    positions = {label: np.flatnonzero(groups == label) for label in labels}
    unit_ratios = []
    for label in labels:
        index = positions[label]
        fallback_error = np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
        candidate_error = np.sqrt(np.mean(
            (candidates[best_index, index] - y[index]) ** 2
        ))
        unit_ratios.append(candidate_error / max(fallback_error, 1e-12))
    unit_ratios = np.asarray(unit_ratios)
    win_fraction = float(np.mean(unit_ratios < 1.0))
    worst_ratio = float(np.max(unit_ratios))
    if win_fraction < min_unit_win_fraction:
        return PriorTrustDecision(
            False, None, gain, (0.0, 0.0), win_fraction, worst_ratio,
            "prior does not win on enough validation units",
        )
    if worst_ratio > max_worst_unit_rmse_ratio:
        return PriorTrustDecision(
            False, None, gain, (0.0, 0.0), win_fraction, worst_ratio,
            "prior violates worst-unit noninferiority",
        )
    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap_replicates, dtype=np.float64)
    chosen_prediction = candidates[best_index]
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(labels, len(labels), replace=True)
        index = np.concatenate([positions[label] for label in sampled])
        fallback_error = np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
        candidate_error = np.sqrt(np.mean((chosen_prediction[index] - y[index]) ** 2))
        differences[replicate] = fallback_error - candidate_error
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(differences, (alpha, 1.0 - alpha))
    interval = (float(low), float(high))
    if low <= 0:
        return PriorTrustDecision(False, None, gain, interval, win_fraction, worst_ratio,
                                  "unit-bootstrap improvement is not positive")
    return PriorTrustDecision(True, best_index, gain, interval, win_fraction, worst_ratio,
                              "validation gain replicated across physical units")
