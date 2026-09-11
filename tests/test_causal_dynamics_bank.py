import numpy as np

from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)


def fixture():
    rng = np.random.default_rng(7)

    def rows(labels, starts):
        correction, context, y, groups, anchor = [], [], [], [], []
        for label, start in zip(labels, starts):
            health = start + np.arange(14) * 0.02
            health += rng.normal(0, 0.001, len(health))
            for index in range(2, 13):
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

    return (
        rows(["a", "b", "c", "d"], [1.0, 1.1, 1.2, 1.3]),
        rows(["e", "f", "g"], [1.05, 1.15, 1.25]),
    )


def test_bank_improves_smooth_causal_trajectory():
    train, validation = fixture()
    model = fit_causal_dynamics_bank(
        *train, *validation, mass_grid=np.linspace(0, 1, 21)
    )
    prediction, evidence = predict_causal_dynamics_bank(
        model, validation[0], validation[1], validation[4]
    )
    baseline = np.mean((validation[2] - validation[4]) ** 2)
    assert np.mean((validation[2] - prediction) ** 2) < baseline
    assert np.any(evidence["active"])


def test_bank_exactly_falls_back_outside_context_support():
    train, validation = fixture()
    model = fit_causal_dynamics_bank(*train, *validation)
    shifted = validation[1].copy()
    shifted[:, 0] += 1000
    anchor = np.full(len(shifted), 0.123456789)
    prediction, evidence = predict_causal_dynamics_bank(
        model, validation[0], shifted, anchor
    )
    assert np.array_equal(prediction, anchor)
    assert np.all(evidence["support_rejected"])


def test_bank_requires_three_training_units():
    train, validation = fixture()
    keep = np.isin(train[3], ["a", "b"])
    reduced = tuple(
        value[keep] for value in train
    )
    try:
        fit_causal_dynamics_bank(*reduced, *validation)
    except ValueError as error:
        assert "three train groups" in str(error)
    else:
        raise AssertionError("expected a physical-unit count error")
