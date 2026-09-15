import numpy as np

from pp_extrapolation.paper_ppx import (
    PPXCandidateEvidence,
    admissible_executors,
    select_paper_ppx_from_train,
)
from pp_extrapolation.ppx_contract_inference import detect_optional_executors
from pp_extrapolation.transferability_gate import PriorEvidence


def _candidates():
    return (
        PPXCandidateEvidence("direct_fallback", 10.0, 0.0, 1.0),
        PPXCandidateEvidence("unbounded", 8.0, 0.8, 1.0),
        PPXCandidateEvidence("bounded", 7.5, 0.8, 1.0),
        PPXCandidateEvidence("dual_scale", 3.0, 0.9, 1.0),
        PPXCandidateEvidence("history", 3.5, 0.9, 1.0),
    )


def test_history_on_dual_off_variable_length():
    train = {
        "groups": np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"]),
        "cycles": np.array([1, 2, 3, 1, 2, 3, 4, 5, 6], dtype=float),
        "y": np.ones(9),
    }
    detected = detect_optional_executors(train)
    assert detected.history is True
    assert detected.dual_scale is False

    auto = select_paper_ppx_from_train(
        train, PriorEvidence(False, 1, 0), _candidates()
    )
    assert auto.contract.causal_history
    assert "history" in admissible_executors(auto.contract)
    assert "dual_scale" not in admissible_executors(auto.contract)
    assert "dual_scale" in auto.dropped
    assert auto.decision.executor == "history"


def test_history_on_dual_on_shifted_support():
    groups = np.repeat(["u1", "u2", "u3"], 4)
    cycles = np.tile(np.arange(4, dtype=float), 3) + np.repeat([0.0, 20.0, 40.0], 4)
    health = np.concatenate(
        [np.linspace(1.0, 0.85, 4), np.linspace(0.70, 0.55, 4), np.linspace(0.40, 0.25, 4)]
    )
    train = {
        "groups": groups,
        "cycles": cycles,
        "x": np.column_stack((health, cycles)),
        "y": np.ones(12),
    }
    detected = detect_optional_executors(train)
    assert detected.history is False
    assert detected.dual_scale is True
    assert detected.support_heterogeneity >= 0.5

    auto = select_paper_ppx_from_train(
        train, PriorEvidence(False, 1, 0), _candidates()
    )
    assert "history" in auto.dropped
    assert "dual_scale" in admissible_executors(auto.contract)
    assert auto.decision.executor == "dual_scale"


def test_history_off_snapshot_rows():
    train = {
        "groups": np.array(["a", "b", "c"]),
        "x": np.array([[0.1, 1.0], [0.8, 1.0], [1.5, 1.0]]),
        "y": np.ones(3),
    }
    detected = detect_optional_executors(train)
    assert detected.history is False
    assert detected.dual_scale is True

    auto = select_paper_ppx_from_train(
        train,
        PriorEvidence(True, 1, 0),
        _candidates(),
        boundary_value=0.8,
    )
    assert "history" in auto.dropped
    assert "dual_scale" not in auto.dropped
    assert auto.decision.executor == "dual_scale"


def test_regime_inferred_from_x_without_named_column():
    groups = np.repeat(["u1", "u2", "u3", "u4"], 3)
    cycles = np.tile(np.arange(1, 4, dtype=float), 4)
    health = np.tile(np.linspace(1.0, 0.94, 3), 4)
    protocol = np.repeat([1.0, 1.0, 2.0, 2.0], 3)
    train = {
        "groups": groups,
        "cycles": cycles,
        "x": np.column_stack((health, cycles, protocol)),
        "y": np.ones(12),
    }
    detected = detect_optional_executors(train)
    assert detected.transport is True
    assert "x[2]" in detected.reasons["transport"]


def test_cycles_in_x_are_not_treated_as_regime():
    train = {
        "groups": np.repeat(["c1", "c2", "c3"], 4),
        "cycles": np.tile(np.arange(1, 5, dtype=float), 3),
        "x": np.column_stack(
            (
                np.tile(np.linspace(1.0, 0.90, 4), 3),
                np.tile(np.arange(1, 5, dtype=float), 3),
            )
        ),
        "y": np.ones(12),
    }
    detected = detect_optional_executors(train)
    assert detected.transport is False
    assert detected.history is False


def test_time_varying_code_in_x_opens_history_not_transport():
    groups = np.repeat(["e1", "e2", "e3"], 6)
    cycles = np.tile(np.arange(1, 7, dtype=float), 3)
    health = np.tile(np.linspace(1.0, 0.90, 6), 3)
    tra = np.tile([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 3)
    train = {
        "groups": groups,
        "cycles": cycles,
        "x": np.column_stack((health, cycles, tra)),
        "y": np.ones(18),
    }
    detected = detect_optional_executors(train)
    assert detected.transport is False
    assert detected.history is True


def test_both_off_single_snapshot_unit():
    train = {
        "groups": np.array(["only"]),
        "x": np.array([[0.5, 1.0]]),
        "y": np.array([1.0]),
    }
    detected = detect_optional_executors(train)
    assert detected.history is False
    assert detected.dual_scale is False

    auto = select_paper_ppx_from_train(
        train,
        PriorEvidence(True, 1, 0),
        _candidates(),
        boundary_value=0.8,
    )
    assert "history" in auto.dropped
    assert "dual_scale" in auto.dropped
    assert auto.decision.executor == "bounded"
