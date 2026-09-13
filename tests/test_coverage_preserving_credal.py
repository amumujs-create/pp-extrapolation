import numpy as np

from pp_extrapolation.coverage_preserving_credal import (
    fit_credal_conformal_ppx,
    legacy_candidate_interval,
    predict_credal_conformal_ppx,
)


def sample(candidate_error):
    groups = np.repeat(np.arange(8), 10)
    y = np.tile(np.linspace(1.0, 2.0, 10), 8)
    fallback = y + 1.0
    candidate = y + candidate_error
    distance = np.linspace(0.0, 2.0, len(y))
    return y, groups, fallback, candidate, distance


def test_approved_pp_replays_candidate_and_legacy_interval():
    y, groups, fallback, candidate, distance = sample(0.2)
    model = fit_credal_conformal_ppx(
        y, groups, fallback, candidate, distance, seed=7
    )
    prediction = predict_credal_conformal_ppx(
        model, fallback, candidate, distance
    )
    legacy_lower, legacy_upper = legacy_candidate_interval(
        model, candidate, distance
    )
    assert model.approved
    np.testing.assert_array_equal(prediction.point, candidate)
    np.testing.assert_allclose(prediction.lower, legacy_lower)
    np.testing.assert_allclose(prediction.upper, legacy_upper)


def test_falsified_pp_uses_exact_fallback_and_contains_legacy_interval():
    y, groups, fallback, candidate, distance = sample(2.0)
    model = fit_credal_conformal_ppx(
        y, groups, fallback, candidate, distance, seed=7
    )
    prediction = predict_credal_conformal_ppx(
        model, fallback, candidate, distance
    )
    legacy_lower, legacy_upper = legacy_candidate_interval(
        model, candidate, distance
    )
    assert not model.approved
    np.testing.assert_array_equal(prediction.point, fallback)
    assert np.all(prediction.lower <= legacy_lower)
    assert np.all(prediction.upper >= legacy_upper)


def test_interval_nesting_prevents_coverage_loss():
    y, groups, fallback, candidate, distance = sample(2.0)
    model = fit_credal_conformal_ppx(
        y, groups, fallback, candidate, distance, seed=7
    )
    prediction = predict_credal_conformal_ppx(
        model, fallback, candidate, distance
    )
    legacy_lower, legacy_upper = legacy_candidate_interval(
        model, candidate, distance
    )
    legacy_covered = (y >= legacy_lower) & (y <= legacy_upper)
    updated_covered = (y >= prediction.lower) & (y <= prediction.upper)
    assert np.all(~legacy_covered | updated_covered)


def test_misaligned_inputs_raise():
    y, groups, fallback, candidate, distance = sample(0.2)
    try:
        fit_credal_conformal_ppx(
            y[:-1], groups, fallback, candidate, distance
        )
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned arrays must raise")
