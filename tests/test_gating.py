import pytest

from pp_extrapolation import select_prior_from_scores


def test_gate_abstains_below_margin():
    decision = select_prior_from_scores(
        {"PP": 0.4, "direction": 0.405, "transport": 0.2},
        minimum_gain=0.01,
    )
    assert decision.selected == "PP"
    assert decision.best_prior == "direction"


def test_gate_enables_prior_at_margin():
    decision = select_prior_from_scores(
        {"PP": 0.4, "direction": 0.39, "transport": 0.41},
        minimum_gain=0.01,
    )
    assert decision.selected == "transport"


def test_gate_rejects_invalid_scores():
    with pytest.raises(ValueError):
        select_prior_from_scores({"PP": 0.4, "transport": float("nan")})

