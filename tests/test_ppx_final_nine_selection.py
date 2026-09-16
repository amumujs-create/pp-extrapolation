import numpy as np

from pp_extrapolation.paper_ppx import admissible_executors, select_ppx_from_structure
from pp_extrapolation.ppx_final_nine import (
    FINAL_NINE_CARDS,
    detect_final_nine,
    replay_final_nine_selection,
)


def test_transport_on_for_hust_and_matr_b2_only():
    on = {card.name for card in FINAL_NINE_CARDS if detect_final_nine(card.name).transport}
    assert on == {"HUST", "MATR-b2"}


def test_time_varying_tra_does_not_open_transport():
    detected = detect_final_nine("N-CMAPSS")
    assert detected.history is True
    assert detected.transport is False


def test_dual_on_for_mich_only():
    on = {card.name for card in FINAL_NINE_CARDS if detect_final_nine(card.name).dual_scale}
    assert on == {"MICH"}


def test_final_paper_does_not_treat_first_feature_as_time():
    groups = np.array(["u1"] * 3 + ["u2"] * 6)
    health = np.concatenate(
        (np.linspace(1.0, 0.8, 3), np.linspace(1.0, 0.8, 6))
    )
    auto = select_ppx_from_structure(
        {"groups": groups, "x": health[:, None], "y": np.ones(9)}
    )
    assert not auto.contract.ordered_progression
    assert not auto.contract.causal_history
    assert "history" in auto.dropped


def test_final_paper_opens_history_with_explicit_time_key():
    groups = np.array(["u1"] * 3 + ["u2"] * 6)
    health = np.concatenate(
        (np.linspace(1.0, 0.8, 3), np.linspace(1.0, 0.8, 6))
    )
    timestamp = np.concatenate((np.arange(3), np.arange(6))).astype(float)
    auto = select_ppx_from_structure(
        {
            "groups": groups,
            "x": health[:, None],
            "timestamp": timestamp,
            "y": np.ones(9),
        },
        time_key="timestamp",
    )
    assert auto.contract.ordered_progression
    assert auto.contract.causal_history
    assert auto.decision.executor == "history"


def test_final_nine_executor_selection_matches():
    selected = {}
    for card in FINAL_NINE_CARDS:
        auto = replay_final_nine_selection(card.name)
        selected[card.name] = auto.decision.executor
        allowed = admissible_executors(auto.contract)
        if card.final_executor == "regime_transport":
            assert "regime_transport" in allowed
        if card.final_executor == "dual_scale":
            assert "dual_scale" in allowed
        if card.final_executor == "history":
            assert "history" in allowed
        assert auto.decision.executor == card.final_executor
    assert selected == {
        "HUST": "regime_transport",
        "Sunwoda": "bounded",
        "N-CMAPSS": "history",
        "Virkler": "unbounded",
        "RWTH": "bounded",
        "MATR-b2": "regime_transport",
        "MICH": "dual_scale",
        "NASA": "history",
        "MATR19": "unbounded",
    }
