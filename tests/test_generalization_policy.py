import numpy as np

from pp_extrapolation import select_prior_trust


def test_rejects_small_validation_gain():
    y = np.arange(12, dtype=float)
    groups = np.repeat(["a", "b", "c"], 4)
    fallback = y + 1.0
    candidate = y + 0.99
    decision = select_prior_trust(y, groups, fallback, candidate[None, :],
                                  bootstrap_replicates=200)
    assert not decision.accepted
    assert decision.candidate_index is None


def test_accepts_replicated_unit_gain():
    y = np.arange(12, dtype=float)
    groups = np.repeat(["a", "b", "c"], 4)
    fallback = y + 2.0
    candidate = y + 0.5
    decision = select_prior_trust(y, groups, fallback, candidate[None, :],
                                  bootstrap_replicates=200)
    assert decision.accepted
    assert decision.candidate_index == 0
    assert decision.bootstrap_ci[0] > 0


def test_rejects_too_few_validation_units():
    y = np.arange(8, dtype=float)
    groups = np.repeat(["a", "b"], 4)
    decision = select_prior_trust(
        y, groups, y + 2.0, np.asarray([y + 0.1]), bootstrap_replicates=200,
    )
    assert not decision.accepted
    assert decision.reason == "fewer than three validation units"
