import numpy as np

from pp_extrapolation.falsification_authority_head import (
    apply_unit_authority,
    fit_falsification_authority_head,
    predict_regret_upper,
    predict_residual_authority,
)


def sample():
    domains = np.repeat(["a", "b", "c", "d"], 8)
    x = np.linspace(-2, 2, len(domains))
    features = np.column_stack([x, x**2])
    regret = 0.4 * x + np.tile(np.linspace(-0.05, 0.05, 8), 4)
    return features, regret, domains


def test_head_fits_with_domain_loo_calibration():
    model = fit_falsification_authority_head(*sample())
    assert model.ridge_alpha > 0
    assert len(model.training_domains) == 4
    assert model.inner_oof_upper_coverage >= 0.90


def test_authority_is_bounded_and_decreases_with_upper_regret():
    model = fit_falsification_authority_head(*sample())
    features = np.asarray([[-2.0, 4.0], [2.0, 4.0]])
    authority, upper = predict_residual_authority(model, features)
    assert np.all((authority >= 0) & (authority <= 1))
    assert upper[0] < upper[1]
    assert authority[0] >= authority[1]


def test_positive_upper_regret_has_zero_authority():
    model = fit_falsification_authority_head(*sample())
    features = np.asarray([[100.0, 10_000.0]])
    authority, upper = predict_residual_authority(model, features)
    assert upper[0] > 0
    assert authority[0] == 0


def test_unit_authority_has_exact_fallback_at_zero():
    fallback = np.asarray([1.0, 2.0, 3.0, 4.0])
    candidate = fallback + 10
    groups = np.asarray(["a", "a", "b", "b"])
    prediction = apply_unit_authority(
        fallback,
        candidate,
        groups,
        np.asarray(["a", "b"]),
        np.asarray([0.0, 0.5]),
    )
    np.testing.assert_array_equal(prediction[:2], fallback[:2])
    np.testing.assert_allclose(prediction[2:], fallback[2:] + 5)


def test_bad_feature_shape_raises():
    model = fit_falsification_authority_head(*sample())
    try:
        predict_regret_upper(model, np.ones((3, 1)))
    except ValueError:
        pass
    else:
        raise AssertionError("bad features must raise")
