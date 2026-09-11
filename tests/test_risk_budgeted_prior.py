import numpy as np

from pp_extrapolation.risk_budgeted_prior import (
    apply_prior_residual,
    fit_support_scale,
    local_budget_modulation,
    select_risk_budget,
    support_distance,
)


def problem():
    groups = np.repeat(np.arange(10), 5)
    y = np.linspace(0, 5, len(groups))
    return y, groups


def test_harmful_prior_recovers_exact_baseline():
    y, groups = problem()
    baseline = y + 0.2
    harmful = y + 3.0
    decision = select_risk_budget(y, groups, baseline, harmful, epsilon=0.0)
    assert decision.alpha == 0
    assert np.array_equal(
        apply_prior_residual(baseline, harmful, decision.alpha), baseline
    )


def test_consistently_helpful_prior_receives_full_budget():
    y, groups = problem()
    baseline = y + 1.0
    prior = y + 0.1
    decision = select_risk_budget(y, groups, baseline, prior, epsilon=0.0)
    assert decision.alpha == 1
    assert decision.mean_excess_ratio < 0
    assert decision.tail_excess_ratio < 0


def test_tail_constraint_reduces_budget_beyond_mean_only():
    y, groups = problem()
    baseline = y + 1.0
    prior = y + 0.2
    prior[groups == 9] = y[groups == 9] + 4.0
    mean_only = select_risk_budget(
        y, groups, baseline, prior, epsilon=0.0, enforce_tail=False
    )
    tail = select_risk_budget(
        y, groups, baseline, prior, epsilon=0.0, enforce_tail=True
    )
    assert tail.alpha < mean_only.alpha


def test_local_modulation_decreases_with_risk_evidence():
    train = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    distance = support_distance(
        np.array([[0.5, 0.5], [2.0, 0.5]]), fit_support_scale(train)
    )
    modulation = local_budget_modulation(
        distance, np.array([0.1, 1.0]), distance_scale=1.0,
        disagreement_scale=1.0,
    )
    assert distance[0] == 0
    assert distance[1] > 0
    assert 0 < modulation[1] < modulation[0] <= 1
