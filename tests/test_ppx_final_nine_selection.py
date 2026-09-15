from pp_extrapolation.paper_ppx import admissible_executors
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
