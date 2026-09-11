import numpy as np
import pytest

from pp_extrapolation import (
    apply_adaptive_shrinkage,
    select_adaptive_prior_shrinkage,
)


def grouped_problem():
    y = np.arange(12, dtype=float)
    groups = np.repeat(np.arange(4), 3)
    fallback = y + 2.0
    return y, groups, fallback


def test_zero_weight_is_exact_fallback():
    fallback = np.array([1.0, 2.0, 3.0])
    candidate = np.array([4.0, 5.0, 6.0])
    assert np.array_equal(
        apply_adaptive_shrinkage(fallback, candidate, 0.0), fallback
    )


def test_strong_replicated_candidate_gets_full_weight():
    y, groups, fallback = grouped_problem()
    candidates = np.stack([y + 0.2, y + 0.5])
    decision = select_adaptive_prior_shrinkage(
        y, groups, fallback, candidates, np.array([0.1, 0.2]),
        bootstrap_replicates=500,
    )
    assert decision.evidence_level == "strong"
    assert decision.prior_weight == 1.0
    assert decision.candidate_index == 0


def test_directional_but_uncertain_candidate_is_shrunk():
    y = np.zeros(12)
    groups = np.repeat(np.arange(4), 3)
    fallback = np.tile([1.0, 1.0, 1.0], 4)
    candidate = fallback.copy()
    candidate[groups < 3] -= 0.03
    candidate[groups == 3] += 0.06
    decision = select_adaptive_prior_shrinkage(
        y, groups, fallback, candidate[None, :], np.array([0.2]),
        strong_relative_gain=0.02, bootstrap_replicates=500,
    )
    assert decision.evidence_level == "weak"
    assert 0 < decision.prior_weight <= 0.5


def test_nonpositive_candidate_returns_exact_fallback():
    y, groups, fallback = grouped_problem()
    decision = select_adaptive_prior_shrinkage(
        y, groups, fallback, (fallback + 1)[None, :], np.array([0.2]),
        bootstrap_replicates=500,
    )
    assert decision.evidence_level == "none"
    assert decision.prior_weight == 0.0


def test_invalid_blend_weight_rejected():
    with pytest.raises(ValueError):
        apply_adaptive_shrinkage(np.ones(2), np.ones(2), 1.1)
