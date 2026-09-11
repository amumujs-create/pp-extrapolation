import numpy as np

from pp_extrapolation.continuous_portfolio import (
    combine_portfolio,
    select_continuous_portfolio,
)


def problem():
    groups = np.repeat(np.arange(6), 4)
    y = np.linspace(0, 10, len(groups))
    return y, groups


def test_invalid_prior_gets_baseline_weight():
    y, groups = problem()
    baseline = y + 0.5
    harmful = y + 3.0
    decision = select_continuous_portfolio(
        y, groups, np.stack([baseline, harmful]), np.array([0.0, 0.2]),
    )
    assert decision.baseline_weight > 0.99
    assert decision.total_prior_weight < 0.01


def test_consistent_prior_receives_continuous_weight():
    y, groups = problem()
    baseline = y + 1.0
    prior = y + 0.3
    decision = select_continuous_portfolio(
        y, groups, np.stack([baseline, prior]), np.array([0.0, 0.2]),
    )
    assert 0 < decision.total_prior_weight <= 1
    prediction = combine_portfolio(
        np.stack([baseline, prior]), np.asarray(decision.weights)
    )
    assert np.mean((prediction - y) ** 2) < np.mean((baseline - y) ** 2)


def test_one_harmful_group_reduces_prior_mass():
    y, groups = problem()
    baseline = y + 1.0
    safe = y + 0.5
    risky = safe.copy()
    risky[groups == 5] = y[groups == 5] + 5.0
    safe_decision = select_continuous_portfolio(
        y, groups, np.stack([baseline, safe]), np.array([0.0, 0.2]),
    )
    risky_decision = select_continuous_portfolio(
        y, groups, np.stack([baseline, risky]), np.array([0.0, 0.2]),
    )
    assert risky_decision.total_prior_weight < safe_decision.total_prior_weight


def test_row_duplication_does_not_change_group_balanced_weights():
    y, groups = problem()
    experts = np.stack([y + 1.0, y + 0.4])
    first = select_continuous_portfolio(
        y, groups, experts, np.array([0.0, 0.2])
    )
    second = select_continuous_portfolio(
        np.repeat(y, 2), np.repeat(groups, 2),
        np.repeat(experts, 2, axis=1), np.array([0.0, 0.2]),
    )
    assert np.allclose(first.weights, second.weights, atol=1e-5)
