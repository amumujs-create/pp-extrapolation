import numpy as np

from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    group_balanced_energy_score,
    predict_residual_mean,
    predict_residual_quantile,
    sample_residual_transport,
    select_transport_rho,
    support_modulation,
)


def _problem(harmful=False):
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(8), 10)
    x = rng.uniform(-1, 1, (len(groups), 2))
    location = 2.0 + 0.5 * x[:, 0]
    residual = 0.35 * x[:, 1] + rng.normal(0, 0.08, len(x))
    y = location + residual
    envelope = ContractEnvelope(location, np.full(len(x), 0.8))

    validation_groups = np.repeat(np.arange(8), 5)
    validation_x = rng.uniform(-1, 1, (len(validation_groups), 2))
    validation_location = 2.0 + 0.5 * validation_x[:, 0]
    validation_y = validation_location + 0.35 * validation_x[:, 1]
    validation_envelope = ContractEnvelope(
        validation_location, np.full(len(validation_x), 0.8)
    )
    baseline = np.repeat(validation_location[:, None], 64, axis=1)
    if harmful:
        validation_y = validation_location - 0.75
    return (
        x, y, groups, envelope, validation_x, validation_y,
        validation_groups, validation_envelope, baseline,
    )


def test_contract_is_frozen_and_all_samples_respect_envelope():
    data = _problem()
    model = fit_residual_transport(*data)
    samples = sample_residual_transport(
        model, data[4], data[7], data[8], seed=9
    )
    assert np.all(samples >= data[7].location[:, None] - data[7].radius[:, None])
    assert np.all(samples <= data[7].location[:, None] + data[7].radius[:, None])
    original = data[7].location.copy()
    with np.testing.assert_raises(ValueError):
        data[7].location[0] = 99
    assert np.array_equal(data[7].location, original)


def test_rho_zero_is_bitwise_exact_distributional_fallback():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    groups = np.arange(4)
    baseline = np.zeros((4, 16))
    harmful = np.full((4, 16), 10.0)
    decision = select_transport_rho(
        y, groups, baseline, harmful,
        mean_budget=0.0, cvar_budget=0.0, max_budget=0.0,
    )
    assert decision.rho == 0.0

    data = _problem()
    fitted = fit_residual_transport(*data)
    rejected = fitted.__class__(
        **{**fitted.__dict__, "rho": 0.0}
    )
    returned = sample_residual_transport(
        rejected, data[4], data[7], data[8], seed=123
    )
    assert np.array_equal(returned, data[8])


def test_monotone_transport_and_mean_quantile_api():
    data = _problem()
    model = fit_residual_transport(*data)
    first = sample_residual_transport(
        model, data[4], data[7], data[8], seed=11
    )
    second = sample_residual_transport(
        model, data[4], data[7], data[8], seed=11
    )
    assert np.array_equal(first, second)
    mean = predict_residual_mean(
        model, data[4], data[7], data[8], seed=11
    )
    quantile = predict_residual_quantile(
        model, data[4], data[7], data[8], [0.1, 0.9], seed=11
    )
    assert mean.shape == (len(data[4]),)
    assert quantile.shape == (2, len(data[4]))
    assert np.all(quantile[0] <= quantile[1])


def test_support_modulation_decreases_outside_train_support():
    data = _problem()
    model = fit_residual_transport(*data)
    x = np.array([[0.0, 0.0], [10.0, 10.0]])
    modulation = support_modulation(x, np.array([0.0, 1.0]), model)
    assert 0 < modulation[1] < modulation[0] <= 1


def test_group_balanced_energy_is_unit_macro_not_row_macro():
    y = np.zeros(11)
    groups = np.array([0] * 10 + [1])
    samples = np.zeros((11, 4))
    samples[-1] = 2.0
    assert group_balanced_energy_score(y, samples, groups) == 1.0
