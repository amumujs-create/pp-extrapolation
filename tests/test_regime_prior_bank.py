import numpy as np

from pp_extrapolation.regime_prior_bank import (
    fit_auto_regime_prior,
    fit_regime_map,
    predict_auto_regime_prior,
    regime_probabilities,
)


def test_train_only_regime_probabilities_are_normalized():
    train_x = np.r_[np.full((20, 2), -2.0), np.full((20, 2), 2.0)]
    model = fit_regime_map(train_x, n_regimes=2)
    probability = regime_probabilities(
        model, np.array([[-2.0, -2.0], [2.0, 2.0]])
    )
    assert probability.shape == (2, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.argmax(probability[0]) != np.argmax(probability[1])


def test_harmful_bank_returns_exact_baseline():
    train_x = np.r_[np.full((30, 1), -2.0), np.full((30, 1), 2.0)]
    groups = np.repeat(np.arange(12), 3)
    validation_x = np.repeat(
        np.array([[-2.0], [2.0]]), len(groups) // 2, axis=0
    )
    y = np.linspace(0, 3, len(groups))
    baseline = y + 0.2
    priors = np.stack([y + 2.0, y + 3.0])
    decision = fit_auto_regime_prior(
        train_x, validation_x, y, groups, baseline, priors,
        np.array([0.02, 0.05]), n_regimes=2,
    )
    prediction = predict_auto_regime_prior(
        decision, validation_x, baseline, priors, np.array([0.02, 0.05])
    )
    assert decision.common_scale == 0
    assert np.array_equal(prediction, baseline)


def test_regimes_can_choose_different_weak_priors():
    train_x = np.r_[np.full((50, 1), -3.0), np.full((50, 1), 3.0)]
    groups = np.repeat(np.arange(20), 2)
    validation_x = np.r_[
        np.full((20, 1), -3.0), np.full((20, 1), 3.0)
    ]
    y = np.linspace(0, 4, len(groups))
    baseline = y + 1.0
    first = y + 3.0
    second = y + 3.0
    first[:20] = y[:20] + 0.1
    second[20:] = y[20:] + 0.1
    priors = np.stack([first, second])
    trusts = np.array([0.02, 0.05])
    decision = fit_auto_regime_prior(
        train_x, validation_x, y, groups, baseline, priors, trusts,
        n_regimes=2,
    )
    selected = {
        item.selected_trust for item in decision.regime_decisions
        if item.accepted
    }
    prediction = predict_auto_regime_prior(
        decision, validation_x, baseline, priors, trusts
    )
    assert selected == {0.02, 0.05}
    assert decision.common_scale > 0
    assert np.mean((prediction - y) ** 2) < np.mean((baseline - y) ** 2)
