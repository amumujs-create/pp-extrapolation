import numpy as np

from pp_extrapolation.predictor_portfolio import (
    fit_predictor_portfolio,
    predict_predictor_portfolio,
)


def fixture():
    rng = np.random.default_rng(11)

    def rows(labels, offset):
        x, y, groups = [], [], []
        for number, label in enumerate(labels):
            coordinate = np.linspace(0.0, 1.0, 24)
            feature = coordinate + 0.1 * number + offset
            x.append(np.column_stack([
                feature,
                coordinate,
                np.sin(2.0 * coordinate),
            ]))
            y.append(0.3 + 1.7 * feature + 0.2 * coordinate**2
                     + rng.normal(0, 0.005, len(feature)))
            groups.extend([label] * len(feature))
        return np.concatenate(x), np.concatenate(y), np.asarray(groups)

    return rows(["a", "b", "c"], 0.0), rows(["d", "e"], 0.05)


def test_portfolio_selects_finite_convex_anchor():
    train, validation = fixture()
    model = fit_predictor_portfolio(
        *train, *validation, include_engression=False, seeds=(42,)
    )
    prediction = predict_predictor_portfolio(model, validation[0])
    assert np.isfinite(prediction).all()
    assert np.isclose(model.weights.sum(), 1.0)
    assert np.all(model.weights >= 0)
    persistence_mse = np.mean(
        (validation[1] - validation[0][:, 0]) ** 2
    )
    assert np.mean((validation[1] - prediction) ** 2) < persistence_mse


def test_portfolio_requires_multiple_training_groups():
    train, validation = fixture()
    groups = np.repeat("one", len(train[2]))
    try:
        fit_predictor_portfolio(
            train[0], train[1], groups, *validation,
            include_engression=False, seeds=(42,),
        )
    except ValueError as error:
        assert "two training groups" in str(error)
    else:
        raise AssertionError("expected a physical-unit error")


def test_portfolio_restores_persistence_on_absolute_context_shift():
    train, validation = fixture()
    model = fit_predictor_portfolio(
        *train, *validation, include_engression=False, seeds=(42,)
    )
    shifted = validation[0].copy()
    shifted[:, 1:] += 1000.0
    prediction = predict_predictor_portfolio(model, shifted)
    assert np.array_equal(prediction, shifted[:, 0])
