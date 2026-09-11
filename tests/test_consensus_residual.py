import numpy as np

from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.stability_first import raw_unit_regret


def sample(target_residual=True):
    train_x = np.tile(np.linspace(-1, 1, 12), 6)[:, None]
    train_groups = np.repeat(np.arange(6), 12)
    train_anchor = np.zeros(len(train_x))
    train_y = 0.1 + 0.2 * train_x[:, 0]
    validation_x = np.tile(np.linspace(-0.9, 0.9, 10), 3)[:, None]
    validation_groups = np.repeat(np.arange(3), 10)
    validation_anchor = np.zeros(len(validation_x))
    if target_residual:
        validation_y = 0.1 + 0.2 * validation_x[:, 0]
    else:
        validation_y = validation_anchor.copy()
    return (
        train_x,
        np.c_[train_x, train_x**2],
        train_y,
        train_groups,
        train_anchor,
        validation_x,
        np.c_[validation_x, validation_x**2],
        validation_y,
        validation_groups,
        validation_anchor,
    )


def test_consensus_residual_improves_supported_validation():
    fit = fit_consensus_residual(*sample())
    assert fit.deployment_mass > 0
    assert fit.validation_macro_improvement > 0
    assert fit.validation_max_regret <= 1e-12


def test_consensus_residual_falls_back_when_anchor_is_exact():
    fit = fit_consensus_residual(*sample(target_residual=False))
    assert fit.deployment_mass == 0
    validation_x = sample()[5]
    anchor = np.arange(len(validation_x), dtype=float)
    prediction, evidence = predict_consensus_residual(
        fit,
        validation_x,
        np.c_[validation_x, validation_x**2],
        anchor,
    )
    np.testing.assert_array_equal(prediction, anchor)
    assert not np.any(evidence["active"])


def test_consensus_residual_rejects_shifted_context_exactly():
    fit = fit_consensus_residual(*sample())
    correction_x = np.zeros((5, 1))
    context_x = np.full((5, 2), 100.0)
    anchor = np.linspace(0.0, 1.0, 5)
    prediction, evidence = predict_consensus_residual(
        fit, correction_x, context_x, anchor
    )
    np.testing.assert_array_equal(prediction, anchor)
    assert np.all(evidence["support_rejected"])


def test_raw_unit_regret_has_no_cross_unit_stabilizer():
    truth = np.zeros(4)
    groups = np.asarray(["a", "a", "b", "b"])
    baseline = np.asarray([1.0, 1.0, 2.0, 2.0])
    prediction = np.asarray([2.0, 2.0, 1.0, 1.0])
    np.testing.assert_allclose(
        raw_unit_regret(truth, groups, baseline, prediction),
        np.asarray([3.0, -0.75]),
    )


def test_consensus_residual_supports_bounded_group_folds():
    fit = fit_consensus_residual(*sample(), max_ensemble_folds=3)
    assert fit.ensemble_folds == 3
    assert fit.coefficients.shape[0] == 3
