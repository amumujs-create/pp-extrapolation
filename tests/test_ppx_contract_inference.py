import numpy as np
import pytest

from pp_extrapolation.ppx_contract_inference import (
    DatasetStructure,
    infer_ppx_contract,
    infer_ppx_contract_from_train_rows,
)


def test_snapshot_rows_disable_history_executor():
    train = {
        "groups": np.array(["a", "b", "c"]),
        "x": np.zeros((3, 4)),
        "y": np.ones(3),
    }
    result = infer_ppx_contract_from_train_rows(train, boundary_value=0.8)
    assert not result.contract.causal_history
    assert not result.contract.ordered_progression
    assert result.contract.known_boundary


def test_equal_length_trajectory_is_not_history_executor():
    groups = np.array(["u1", "u1", "u1", "u2", "u2", "u2"])
    cycles = np.array([1, 2, 3, 10, 11, 12], dtype=float)
    train = {"groups": groups, "cycles": cycles, "y": np.ones(6)}
    result = infer_ppx_contract_from_train_rows(train, boundary_value=0.8)
    assert result.contract.ordered_progression
    assert not result.contract.causal_history
    assert result.contract.known_boundary


def test_variable_length_trajectory_enables_history_executor():
    groups = np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"])
    cycles = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6], dtype=float)
    train = {"groups": groups, "cycles": cycles, "y": np.ones(9)}
    result = infer_ppx_contract_from_train_rows(train)
    assert result.contract.causal_history
    assert "시점 수 비" in result.reasons["causal_history"]


def test_regime_and_support_flags():
    groups = np.repeat(["u1", "u2", "u3", "u4"], 4)
    cycles = np.tile(np.arange(4, dtype=float), 4) + np.repeat([0, 10, 20, 30], 4)
    regimes = np.repeat(["A", "A", "B", "B"], 4)
    result = infer_ppx_contract(
        DatasetStructure(
            unit_ids=groups,
            progression_coordinate=cycles,
            regime_ids=regimes,
        )
    )
    assert result.contract.observed_regime
    assert result.contract.support_heterogeneity_available
    assert result.support_heterogeneity >= 0.5


def test_non_monotonic_coordinate_disables_ordered_progression():
    groups = np.array(["u1", "u1", "u1"])
    cycles = np.array([1.0, 3.0, 2.0])
    result = infer_ppx_contract(
        DatasetStructure(unit_ids=groups, progression_coordinate=cycles)
    )
    assert not result.contract.ordered_progression
    assert not result.contract.causal_history


def test_ds03_like_contract():
    groups = np.repeat(np.arange(8), 5)
    cycles = np.tile(np.arange(5, dtype=float), 8)
    train = {"groups": groups, "cycles": cycles, "y": np.ones(40)}
    result = infer_ppx_contract_from_train_rows(train)
    assert not result.contract.known_boundary
    assert result.contract.ordered_progression
    assert not result.contract.causal_history
    assert not result.contract.observed_regime
    assert not result.contract.support_heterogeneity_available


def test_missing_group_key_raises():
    with pytest.raises(ValueError, match="groups"):
        infer_ppx_contract_from_train_rows({"y": np.ones(3)})


def test_auto_select_drops_history_without_trajectory():
    from pp_extrapolation.paper_ppx import (
        PPXCandidateEvidence,
        select_paper_ppx_from_train,
    )
    from pp_extrapolation.transferability_gate import PriorEvidence

    train = {
        "groups": np.array(["a", "b", "c"]),
        "x": np.zeros((3, 2)),
        "y": np.ones(3),
    }
    auto = select_paper_ppx_from_train(
        train,
        PriorEvidence(True, 1, 0),
        (
            PPXCandidateEvidence("direct_fallback", 10.0, 0.0, 1.0),
            PPXCandidateEvidence("unbounded", 4.0, 0.8, 1.0),
            PPXCandidateEvidence("history", 3.0, 0.9, 1.0),
        ),
        boundary_value=0.8,
    )
    assert not auto.contract.causal_history
    assert "history" in auto.dropped
    assert auto.decision.executor == "unbounded"


def test_auto_select_keeps_history_when_units_have_time():
    from pp_extrapolation.paper_ppx import (
        PPXCandidateEvidence,
        select_paper_ppx_from_train,
    )
    from pp_extrapolation.transferability_gate import PriorEvidence

    train = {
        "groups": np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"]),
        "cycles": np.array([1, 2, 3, 1, 2, 3, 4, 5, 6], dtype=float),
        "y": np.ones(9),
    }
    auto = select_paper_ppx_from_train(
        train,
        PriorEvidence(False, 1, 0),
        (
            PPXCandidateEvidence("direct_fallback", 10.0, 0.0, 1.0),
            PPXCandidateEvidence("history", 4.0, 0.8, 1.0),
        ),
    )
    assert auto.contract.causal_history
    assert auto.dropped == ()
    assert auto.decision.executor == "history"
