import inspect

import pytest

from pp_extrapolation.paper_ppx import (
    PPXCandidateEvidence,
    PPXContract,
    admissible_executors,
    select_paper_ppx,
)
from pp_extrapolation.transferability_gate import PriorEvidence


def contract(**changes):
    values = {
        "known_boundary": True,
        "ordered_progression": True,
        "causal_history": False,
        "observed_regime": False,
        "support_heterogeneity_available": True,
        "fallback": "direct_fallback",
    }
    values.update(changes)
    return PPXContract(**values)


def prior(**changes):
    values = {
        "known_boundary": True,
        "complete_groups": 1,
        "minimum_complete_groups_per_regime": 0,
    }
    values.update(changes)
    return PriorEvidence(**values)


def candidate(name, loss, wins=0.8, worst=1.0):
    return PPXCandidateEvidence(name, loss, wins, worst)


def test_selector_has_no_test_outcome_parameter():
    parameters = inspect.signature(select_paper_ppx).parameters
    assert all("test" not in name and name not in {"y", "target"} for name in parameters)


def test_contract_restricts_executor_candidates():
    allowed = admissible_executors(contract(
        causal_history=False,
        observed_regime=False,
        support_heterogeneity_available=False,
    ))
    assert allowed == (
        "direct_fallback", "prior_only", "unbounded", "bounded"
    )


def test_rejects_contract_inadmissible_candidate():
    with pytest.raises(ValueError, match="contract-inadmissible"):
        select_paper_ppx(
            contract(observed_regime=False),
            prior(),
            (
                candidate("direct_fallback", 10),
                candidate("regime_transport", 5),
            ),
        )


def test_prior_rejection_exactly_selects_fallback():
    decision = select_paper_ppx(
        contract(known_boundary=False),
        prior(
            known_boundary=False,
            complete_groups=3,
            minimum_complete_groups_per_regime=1,
        ),
        (
            candidate("direct_fallback", 10, wins=0, worst=1),
            candidate("unbounded", 4),
        ),
    )
    assert decision.executor == "direct_fallback"
    assert not decision.approved


def test_optional_executor_requires_two_percent_gain():
    decision = select_paper_ppx(
        contract(),
        prior(),
        (
            candidate("direct_fallback", 10, wins=0, worst=1),
            candidate("unbounded", 9.85),
        ),
    )
    assert decision.executor == "direct_fallback"


def test_unit_risk_can_reject_best_average_loss():
    decision = select_paper_ppx(
        contract(),
        prior(),
        (
            candidate("direct_fallback", 10, wins=0, worst=1),
            candidate("unbounded", 8, wins=0.9, worst=1.4),
            candidate("bounded", 9, wins=0.8, worst=1.05),
        ),
    )
    assert decision.executor == "bounded"
    assert decision.approved


def test_lowest_loss_feasible_executor_is_selected():
    decision = select_paper_ppx(
        contract(),
        prior(),
        (
            candidate("direct_fallback", 10, wins=0, worst=1),
            candidate("unbounded", 9),
            candidate("bounded", 8),
            candidate("dual_scale", 7),
        ),
    )
    assert decision.executor == "dual_scale"
