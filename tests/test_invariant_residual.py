import numpy as np

from pp_extrapolation.invariant_residual import (
    fit_invariant_residual,
    predict_invariant_residual,
)


def test_invariant_residual_improves_repeated_small_drift():
    groups = np.repeat(np.arange(8), 4)
    x = np.tile(np.linspace(-1, 1, 4), 8)[:, None]
    anchor = np.ones(len(x))
    y = anchor + 0.01 * x[:, 0]
    model = fit_invariant_residual(
        x, y, groups, anchor, x, y, groups, anchor
    )
    prediction, out_of_support = predict_invariant_residual(model, x, anchor)
    assert not np.any(out_of_support)
    assert model.deployment_mass > 0
    assert np.mean((prediction - y) ** 2) < np.mean((anchor - y) ** 2)


def test_out_of_support_returns_exact_anchor():
    groups = np.repeat(np.arange(6), 3)
    x = np.tile(np.array([-1.0, 0.0, 1.0]), 6)[:, None]
    anchor = np.ones(len(x))
    y = anchor + 0.01 * x[:, 0]
    model = fit_invariant_residual(
        x, y, groups, anchor, x, y, groups, anchor
    )
    test_anchor = np.array([0.9, 1.1])
    prediction, out_of_support = predict_invariant_residual(
        model, np.array([[10.0], [-10.0]]), test_anchor
    )
    assert np.all(out_of_support)
    assert np.array_equal(prediction, test_anchor)
