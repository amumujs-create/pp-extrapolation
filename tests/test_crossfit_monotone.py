import numpy as np

from pp_extrapolation.crossfit_monotone import (
    crossfit_monotone_budget,
    fit_monotone_budget,
    hierarchical_risk_ratios,
    monotone_modulation,
    risk_ratios,
)


def problem():
    groups = np.repeat(np.arange(8), 5)
    y = np.linspace(0, 4, len(groups))
    return y, groups


def test_modulation_is_monotone_in_both_risk_indicators():
    modulation = monotone_modulation(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        distance_decay=1.0,
        disagreement_decay=2.0,
    )
    assert 1.0 == modulation[0]
    assert modulation[0] > modulation[1] > modulation[2] > 0


def test_harmful_prior_has_exact_zero_budget():
    y, groups = problem()
    baseline = y + 0.2
    prior = y + 3.0
    fit = fit_monotone_budget(
        y, groups, baseline, prior, np.zeros_like(y), np.zeros_like(y),
        epsilon=0.0,
    )
    assert fit.alpha == 0.0


def test_crossfit_returns_one_prediction_per_row_and_safe_final_fit():
    y, groups = problem()
    baseline = y + 1.0
    prior = y + 0.2
    result = crossfit_monotone_budget(
        y, groups, baseline, prior, np.linspace(0, 2, len(y)),
        np.linspace(0, 1, len(y)), epsilon=0.0,
    )
    assert result.oof_prediction.shape == y.shape
    assert np.all(np.isfinite(result.oof_prediction))
    assert len(result.fold_parameters) == len(np.unique(groups))
    assert result.final.mean_excess_ratio <= 0
    assert result.final.tail_excess_ratio <= 0


def test_hierarchical_tail_shrinks_one_row_unit_excess():
    groups = np.concatenate([np.repeat(np.arange(5), 10), np.array([5])])
    y = np.zeros(len(groups))
    baseline = np.ones(len(groups))
    prediction = np.zeros(len(groups))
    prediction[-1] = 4.0
    raw_mean, raw_tail = risk_ratios(y, groups, baseline, prediction)
    hierarchical_mean, hierarchical_tail = hierarchical_risk_ratios(
        y, groups, baseline, prediction, shrinkage_n0=5.0
    )
    assert hierarchical_mean == raw_mean
    assert hierarchical_tail < raw_tail
