import numpy as np

from pp_extrapolation import fit_log_boundary_quotient_pp, predict_log_boundary_quotient


def _rows(x, margin, y, groups):
    return {"x": np.asarray(x, float), "margin": np.asarray(margin, float),
            "y": np.asarray(y, float), "groups": np.asarray(groups)}


def test_log_quotient_is_nonnegative_and_exact_at_boundary():
    train = _rows([[0], [1], [2], [3]], [4, 3, 2, 1], [8, 6, 4, 2], ["a"] * 4)
    validation = _rows([[1.5], [2.5]], [2.5, 1.5], [5, 3], ["b"] * 2)
    fit = fit_log_boundary_quotient_pp(
        train, validation, seed=3, selected_alpha=10, max_epochs=2, patience=2
    )
    query = {"x": np.asarray([[2.0], [4.0]]), "margin": np.asarray([2.0, 0.0])}
    prediction = predict_log_boundary_quotient(fit, query)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0)
    assert prediction[1] == 0.0
