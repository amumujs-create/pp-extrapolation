import numpy as np

from pp_extrapolation.boundary_quotient import (
    equal_dataset_unit_weights, fit_boundary_quotient_pp,
    predict_boundary_affine, predict_boundary_quotient,
)


def rows(n=60):
    margin = np.linspace(.05, 1., n).astype("float32")
    x = np.c_[margin, np.log1p(margin)].astype("float32")
    return {"x": x, "margin": margin, "y": (3 * margin).astype("float32"),
            "units": np.arange(n) // max(n // 6, 1),
            "dataset": np.arange(n) >= n // 2}


def test_equal_dataset_unit_weights_balance_both_levels():
    value = rows(); w = equal_dataset_unit_weights(value["dataset"], value["units"])
    assert np.isclose(w.mean(), 1)
    assert np.allclose([w[value["dataset"] == d].sum() for d in (0, 1)], 30)


def test_boundary_quotient_is_exact_zero_and_finite():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(train, validation, seed=1, max_epochs=3, patience=2)
    query = {**validation, "margin": np.zeros(30, dtype="float32")}
    assert np.count_nonzero(predict_boundary_quotient(fit, query)) == 0
    assert np.count_nonzero(predict_boundary_affine(fit, query)) == 0
    assert np.isfinite(predict_boundary_quotient(fit, validation)).all()


def test_bounded_residual_contracts_toward_the_failure_boundary():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(train, validation, seed=2, max_epochs=4, patience=3,
                                   residual_bound=1.5)
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    # Softplus is 1-Lipschitz and the score correction is bounded by B.
    envelope = validation["margin"] * fit.model.residual_bound
    assert np.all(np.abs(prediction - affine) <= envelope + 1e-6)

    half = {**validation, "margin": validation["margin"] / 2}
    delta = np.abs(predict_boundary_quotient(fit, half) - predict_boundary_affine(fit, half))
    assert np.allclose(delta, np.abs(prediction - affine) / 2, atol=1e-6)


def test_unbounded_residual_variant_remains_finite():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(train, validation, seed=3, max_epochs=3,
                                   patience=2, residual_bound=None)
    assert fit.selection["residual_bound"] is None
    assert np.isfinite(predict_boundary_quotient(fit, validation)).all()


def test_margin_adaptive_envelope_still_contracts_to_zero():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(train, validation, seed=4, max_epochs=4, patience=3,
                                   residual_bound=2.0, late_bound_growth=3.0)
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    margin = validation["margin"]
    envelope = margin * 2.0 * (1.0 + 3.0 * (1.0 - np.clip(margin, 0, 1)))
    assert np.all(np.abs(prediction - affine) <= envelope + 1e-6)
    query = {**validation, "margin": np.zeros(len(margin), dtype="float32")}
    assert np.count_nonzero(predict_boundary_quotient(fit, query)) == 0


def test_regime_conditioned_extra_residual_has_global_envelope():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(
        train, validation, seed=5, max_epochs=4, patience=3,
        residual_bound=1.5, extra_residual_bound=3.0, regime_gate_penalty=1e-3,
    )
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    envelope = validation["margin"] * (1.5 + 3.0)
    assert np.all(np.abs(prediction - affine) <= envelope + 1e-6)
    assert fit.selection["extra_residual_bound"] == 3.0
    query = {**validation, "margin": np.zeros(len(validation["margin"]), dtype="float32")}
    assert np.count_nonzero(predict_boundary_quotient(fit, query)) == 0


def test_power_shaped_late_envelope_is_respected():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(
        train, validation, seed=6, max_epochs=4, patience=3,
        residual_bound=2.0, late_bound_growth=3.0, late_bound_power=2.0,
    )
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    margin = validation["margin"]
    envelope = margin * 2.0 * (1.0 + 3.0 * (1.0 - np.clip(margin, 0, 1)) ** 2)
    assert np.all(np.abs(prediction - affine) <= envelope + 1e-6)


def test_support_gated_envelope_keeps_global_contraction_bound():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(
        train, validation, seed=7, max_epochs=4, patience=3,
        residual_bound=2.0, late_bound_growth=3.0, support_gate_feature=1,
        support_gate_threshold=0.5, support_gate_temperature=0.25,
    )
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    envelope = validation["margin"] * 2.0 * (1.0 + 3.0)
    assert np.all(np.abs(prediction - affine) <= envelope + 1e-6)


def test_ema_fit_returns_one_finite_boundary_consistent_model():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(
        train, validation, seed=8, max_epochs=5, patience=4,
        residual_bound=2.0, ema_decay=0.99,
    )
    assert fit.selection["ema_decay"] == 0.99
    assert np.isfinite(predict_boundary_quotient(fit, validation)).all()
    query = {**validation, "margin": np.zeros(len(validation["margin"]), dtype="float32")}
    assert np.count_nonzero(predict_boundary_quotient(fit, query)) == 0


def test_swa_refit_returns_one_finite_model():
    train = rows(60)
    fit = fit_boundary_quotient_pp(
        train, train, seed=9, max_epochs=6, patience=20, restore_best=False,
        residual_bound=2.0, swa_start_fraction=0.5,
    )
    assert fit.selection["swa_checkpoints"] == 4
    assert np.isfinite(predict_boundary_quotient(fit, train)).all()


def test_dual_scale_saturation_has_finite_global_envelope():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(
        train, validation, seed=10, max_epochs=5, patience=4,
        residual_bound=2.0, broad_residual_bound=10.0,
        local_saturation_weight=0.3,
    )
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    envelope = validation["margin"] * (0.3 * 2.0 + 0.7 * 10.0)
    assert np.all(np.abs(prediction - affine) <= envelope + 1e-6)


def test_support_adaptive_dual_scale_is_bounded_by_broad_envelope():
    train = rows(60); validation = rows(30)
    fit = fit_boundary_quotient_pp(
        train, validation, seed=11, max_epochs=5, patience=4,
        residual_bound=2.0, broad_residual_bound=8.0,
        local_saturation_weight=0.2, support_gate_feature=1,
        support_adaptive_saturation=True,
    )
    prediction = predict_boundary_quotient(fit, validation)
    affine = predict_boundary_affine(fit, validation)
    assert np.all(np.abs(prediction - affine) <= validation["margin"] * 8.0 + 1e-6)
