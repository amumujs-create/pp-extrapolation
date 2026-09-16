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


def test_monotonic_first_feature_is_not_implicit_time():
    groups = np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"])
    health = np.array([1.0, 0.9, 0.8, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75])
    train = {
        "groups": groups,
        "x": health[:, None],
        "y": np.ones(len(groups)),
    }
    result = infer_ppx_contract_from_train_rows(train)
    assert not result.contract.ordered_progression
    assert not result.contract.causal_history
    assert "진행 좌표가 없다" in result.reasons["ordered_progression"]


def test_explicit_nonstandard_time_key_enables_time_audit():
    groups = np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"])
    timestamp = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6], dtype=float)
    train = {"groups": groups, "timestamp": timestamp, "y": np.ones(9)}
    result = infer_ppx_contract_from_train_rows(train, time_key="timestamp")
    assert result.contract.ordered_progression
    assert result.contract.causal_history


def test_blank_time_key_auto_detects_one_monotonic_train_column():
    groups = np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"])
    timestamp = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6], dtype=float)
    train = {"groups": groups, "timestamp": timestamp, "y": np.ones(9)}
    result = infer_ppx_contract_from_train_rows(train, time_key="")
    assert result.contract.ordered_progression
    assert result.contract.causal_history
    assert "timestamp" in result.reasons["time_source"]


def test_blank_time_key_rejects_ambiguous_monotonic_columns():
    groups = np.array(["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"])
    timestamp = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6], dtype=float)
    elapsed = timestamp * 10.0
    train = {
        "groups": groups,
        "timestamp": timestamp,
        "elapsed": elapsed,
        "y": np.ones(9),
    }
    result = infer_ppx_contract_from_train_rows(train, time_key="")
    assert not result.contract.ordered_progression
    assert not result.contract.causal_history
    assert "여러 개" in result.reasons["time_source"]


def test_regime_user_key_wins_and_blank_key_auto_detects():
    groups = np.repeat(["u1", "u2", "u3", "u4"], 3)
    timestamp = np.tile(np.arange(3, dtype=float), 4)
    operating_mode = np.repeat(["A", "A", "B", "B"], 3)
    train = {
        "groups": groups,
        "timestamp": timestamp,
        "operating_mode": operating_mode,
        "y": np.ones(12),
    }
    explicit = infer_ppx_contract_from_train_rows(
        train,
        time_key="timestamp",
        regime_key="operating_mode",
    )
    automatic = infer_ppx_contract_from_train_rows(
        train,
        time_key="timestamp",
        regime_key="",
    )
    assert explicit.contract.observed_regime
    assert automatic.contract.observed_regime
    assert "operating_mode" in explicit.reasons["regime_source"]
    assert "operating_mode" in automatic.reasons["regime_source"]


def test_group_user_key_wins_and_blank_key_auto_detects():
    specimen_code = np.repeat(["s1", "s2", "s3", "s4"], 3)
    timestamp = np.tile(np.arange(3, dtype=float), 4)
    train = {
        "specimen_code": specimen_code,
        "timestamp": timestamp,
        "y": np.ones(12),
    }
    explicit = infer_ppx_contract_from_train_rows(
        train,
        group_key="specimen_code",
        time_key="timestamp",
    )
    automatic = infer_ppx_contract_from_train_rows(
        train,
        group_key="",
        time_key="timestamp",
    )
    assert explicit.contract.ordered_progression
    assert automatic.contract.ordered_progression
    assert "specimen_code" in explicit.reasons["group_source"]
    assert "specimen_code" in automatic.reasons["group_source"]


def test_group_auto_detection_prefers_unit_blocks_over_regime_blocks():
    specimen_code = np.repeat(["s1", "s2", "s3", "s4"], 3)
    operating_mode = np.repeat(["A", "A", "B", "B"], 3)
    timestamp = np.tile(np.arange(3, dtype=float), 4)
    train = {
        "specimen_code": specimen_code,
        "operating_mode": operating_mode,
        "timestamp": timestamp,
        "y": np.ones(12),
    }
    result = infer_ppx_contract_from_train_rows(
        train,
        time_key="timestamp",
        regime_key="operating_mode",
    )
    assert "specimen_code" in result.reasons["group_source"]
    assert result.contract.observed_regime


def test_ambiguous_group_columns_require_user_key():
    train = {
        "specimen_a": np.repeat(["a1", "a2"], 3),
        "specimen_b": np.repeat(["b1", "b2"], 3),
        "y": np.ones(6),
    }
    with pytest.raises(ValueError, match="ambiguous"):
        infer_ppx_contract_from_train_rows(train)


def test_unknown_user_column_has_clear_error():
    train = {"groups": np.array(["u1", "u1"]), "y": np.ones(2)}
    with pytest.raises(ValueError, match="requested time column"):
        infer_ppx_contract_from_train_rows(train, time_key="missing_time")
    with pytest.raises(ValueError, match="requested regime column"):
        infer_ppx_contract_from_train_rows(train, regime_key="missing_regime")


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
    with pytest.raises(ValueError, match="group_key"):
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
