import numpy as np

from pp_extrapolation.causal_backtest import (
    apply_adaptive_causal_backtest_gate,
    apply_causal_backtest_gate,
)


def test_gate_activates_after_five_realized_wins():
    origin = np.arange(10)
    target = origin + 1
    truth = np.ones(10)
    anchor = np.zeros(10)
    candidate = np.ones(10)
    evaluate = origin >= 5
    prediction, evidence = apply_causal_backtest_gate(
        candidate,
        anchor,
        truth,
        np.repeat("a", 10),
        origin,
        target,
        evaluate,
    )
    assert not evidence["active"][4]
    assert evidence["active"][5]
    assert prediction[5] == candidate[5]


def test_gate_rejects_if_any_recent_shadow_loses():
    origin = np.arange(10)
    target = origin + 1
    truth = np.ones(10)
    anchor = np.zeros(10)
    candidate = np.ones(10)
    candidate[3] = -1
    evaluate = origin == 5
    prediction, evidence = apply_causal_backtest_gate(
        candidate,
        anchor,
        truth,
        np.repeat("a", 10),
        origin,
        target,
        evaluate,
    )
    assert not evidence["active"][5]
    assert prediction[5] == anchor[5]


def test_gate_does_not_use_unrealized_future_truth():
    origin = np.arange(10)
    target = origin + 2
    truth = np.ones(10)
    anchor = np.zeros(10)
    candidate = np.ones(10)
    evaluate = origin == 7
    first, _ = apply_causal_backtest_gate(
        candidate,
        anchor,
        truth,
        np.repeat("a", 10),
        origin,
        target,
        evaluate,
        required_wins=3,
    )
    changed = truth.copy()
    changed[target > 7] = 1000
    second, _ = apply_causal_backtest_gate(
        candidate,
        anchor,
        changed,
        np.repeat("a", 10),
        origin,
        target,
        evaluate,
        required_wins=3,
    )
    np.testing.assert_array_equal(first, second)


def test_adaptive_gate_activates_with_short_all_winning_history():
    origin = np.arange(6)
    target = origin + 1
    truth = np.ones(6)
    anchor = np.zeros(6)
    candidate = np.ones(6)
    prediction, evidence = apply_adaptive_causal_backtest_gate(
        candidate,
        anchor,
        truth,
        np.repeat("a", 6),
        origin,
        target,
        origin == 2,
        minimum_history=2,
        maximum_history=5,
    )
    assert evidence["available_shadow_count"][2] == 2
    assert evidence["active"][2]
    assert prediction[2] == candidate[2]


def test_adaptive_gate_requires_every_available_shadow_to_win():
    origin = np.arange(6)
    target = origin + 1
    truth = np.ones(6)
    anchor = np.zeros(6)
    candidate = np.ones(6)
    candidate[1] = -1
    prediction, evidence = apply_adaptive_causal_backtest_gate(
        candidate,
        anchor,
        truth,
        np.repeat("a", 6),
        origin,
        target,
        origin == 2,
    )
    assert not evidence["active"][2]
    assert prediction[2] == anchor[2]
