import numpy as np

from pp_extrapolation.risk_budgeted_prior import apply_prior_residual
from pp_extrapolation.stability_first import (
    select_stability_first_prior,
    unit_regret,
)


def problem():
    groups = np.repeat(np.arange(10), 5)
    y = np.linspace(0, 5, len(groups))
    return y, groups


def test_consistently_helpful_prior_is_accepted():
    y, groups = problem()
    baseline = y + 1.0
    priors = np.stack([y + 0.2, y + 0.4])
    decision = select_stability_first_prior(
        y, groups, baseline, priors, np.array([0.02, 0.05])
    )
    assert decision.accepted
    assert decision.selected_trust == 0.02
    assert decision.alpha > 0
    assert decision.fold_prior_rate == 1
    assert decision.modal_trust_agreement == 1


def test_harmful_prior_returns_exact_baseline():
    y, groups = problem()
    baseline = y + 0.2
    priors = np.stack([y + 3.0, y + 4.0])
    decision = select_stability_first_prior(
        y, groups, baseline, priors, np.array([0.02, 0.05])
    )
    assert not decision.accepted
    assert decision.alpha == 0
    assert np.array_equal(
        apply_prior_residual(
            baseline, priors[0], decision.alpha
        ),
        baseline,
    )


def test_one_unstable_unit_prevents_consensus_acceptance():
    y, groups = problem()
    baseline = y + 1.0
    prior = y + 0.2
    prior[groups == 9] = y[groups == 9] + 5.0
    decision = select_stability_first_prior(
        y, groups, baseline, prior[None, :], np.array([0.02])
    )
    assert not decision.accepted


def test_regret_denominator_is_finite_for_near_perfect_unit():
    y, groups = problem()
    baseline = y.copy()
    baseline[groups != 0] += 1.0
    prediction = baseline + 0.1
    regret = unit_regret(y, groups, baseline, prediction)
    assert np.all(np.isfinite(regret))


def test_explicit_low_volume_mode_supports_three_groups():
    groups = np.repeat(np.arange(3), 4)
    y = np.linspace(0, 2, len(groups))
    baseline = y + 1.0
    priors = np.stack([y + 0.2, y + 0.4])
    decision = select_stability_first_prior(
        y, groups, baseline, priors, np.array([0.02, 0.05]),
        min_groups=3,
    )
    assert decision.accepted
    assert decision.fold_prior_rate == 1
