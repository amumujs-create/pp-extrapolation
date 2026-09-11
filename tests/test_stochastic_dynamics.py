import numpy as np

from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
)
from pp_extrapolation.stochastic_dynamics import (
    fit_stochastic_dynamics,
    predict_stochastic_dynamics,
)


def rows(labels):
    correction, context, y, groups, anchor = [], [], [], [], []
    for offset, label in enumerate(labels):
        health = 1.0 + 0.02 * np.arange(12) + offset * 0.01
        for index in range(2, 11):
            slope1 = health[index] - health[index - 1]
            slope2 = (health[index] - health[index - 2]) / 2
            curve = slope1 - slope2
            correction.append([slope1, slope2, curve])
            context.append([
                health[index], slope1, slope2, curve,
                np.mean(health[index - 2:index + 1]),
                np.std(health[index - 2:index + 1]), 1.0,
            ])
            y.append(health[index + 1])
            groups.append(label)
            anchor.append(health[index])
    return tuple(map(np.asarray, (
        correction, context, y, groups, anchor
    )))


def test_stochastic_head_returns_aligned_finite_prediction():
    train = rows(["a", "b", "c", "d", "e"])
    validation = rows(["f", "g", "h"])
    deterministic = fit_causal_dynamics_bank(*train, *validation)
    model = fit_stochastic_dynamics(
        deterministic, *train, *validation
    )
    prediction, evidence = predict_stochastic_dynamics(
        model, validation[0], validation[1], validation[4]
    )
    assert prediction.shape == validation[2].shape
    assert np.all(np.isfinite(prediction))
    assert evidence["active"].shape == prediction.shape


def test_stochastic_head_nonfinite_row_falls_back_exactly():
    train = rows(["a", "b", "c", "d", "e"])
    validation = rows(["f", "g", "h"])
    deterministic = fit_causal_dynamics_bank(*train, *validation)
    model = fit_stochastic_dynamics(
        deterministic, *train, *validation
    )
    context = validation[1].copy()
    context[0, 1] = np.nan
    prediction, evidence = predict_stochastic_dynamics(
        model, validation[0], context, validation[4]
    )
    assert prediction[0] == validation[4][0]
    assert evidence["invalid"][0]
