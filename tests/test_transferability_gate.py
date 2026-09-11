import pytest

from pp_extrapolation.transferability_gate import (
    PriorEvidence,
    select_ppx_route,
)


def evidence(**changes):
    values = {
        "known_boundary": False,
        "complete_groups": 5,
        "minimum_complete_groups_per_regime": 2,
        "oof_prior_regret": -0.1,
        "oof_mode_stability": 0.8,
    }
    values.update(changes)
    return PriorEvidence(**values)


def test_declared_boundary_approves_boundary_route():
    decision = select_ppx_route(evidence(
        known_boundary=True,
        complete_groups=0,
        minimum_complete_groups_per_regime=0,
        oof_prior_regret=None,
        oof_mode_stability=None,
    ))
    assert decision.route == "boundary_pp"
    assert decision.prior_weight == 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"complete_groups": 4},
        {"minimum_complete_groups_per_regime": 1},
        {"oof_prior_regret": None},
        {"oof_prior_regret": 0.01},
        {"oof_mode_stability": None},
        {"oof_mode_stability": 0.59},
    ],
)
def test_missing_or_failed_source_evidence_uses_neural_safety(changes):
    decision = select_ppx_route(evidence(**changes))
    assert decision.route == "neural_safety"
    assert decision.prior_weight == 0.0


def test_complete_stable_source_evidence_approves_transferable_prior():
    decision = select_ppx_route(evidence())
    assert decision.route == "transferable_prior_pp"
    assert decision.prior_weight == 1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_source_evidence_is_rejected(value):
    with pytest.raises(ValueError):
        select_ppx_route(evidence(oof_prior_regret=value))
