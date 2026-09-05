"""Train-only decision rule for optional prior losses."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PriorGateDecision:
    selected: str
    baseline: str
    best_prior: str
    baseline_score: float
    best_prior_score: float
    gain: float
    minimum_gain: float


def select_prior_from_scores(
    scores: Mapping[str, float],
    *,
    baseline: str = "PP",
    minimum_gain: float = 0.01,
) -> PriorGateDecision:
    """Enable a prior only when its pseudo-extrapolation gain clears a margin."""
    if baseline not in scores or len(scores) < 2:
        raise ValueError("scores require the baseline and at least one prior arm")
    if not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum_gain must be finite and nonnegative")
    checked = {str(name): float(score) for name, score in scores.items()}
    if not np.isfinite(list(checked.values())).all():
        raise ValueError("all pseudo-extrapolation scores must be finite")
    # Name is the deterministic secondary key. The baseline wins exact ties.
    prior_scores = {name: score for name, score in checked.items() if name != baseline}
    best_prior = min(prior_scores, key=lambda name: (-prior_scores[name], name))
    gain = prior_scores[best_prior] - checked[baseline]
    selected = best_prior if gain + 1e-12 >= minimum_gain else baseline
    return PriorGateDecision(
        selected=selected,
        baseline=baseline,
        best_prior=best_prior,
        baseline_score=checked[baseline],
        best_prior_score=prior_scores[best_prior],
        gain=gain,
        minimum_gain=float(minimum_gain),
    )
