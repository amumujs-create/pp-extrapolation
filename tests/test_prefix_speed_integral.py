import numpy as np
import pytest
import torch

from pp_extrapolation.prefix_speed_integral import (
    PrefixSpeedIntegral, fit_prefix_speed, make_prefix_pairs, predict_prefix_speed,
)


def model(variable=True):
    return PrefixSpeedIntegral(np.zeros(2), np.ones(2), width=8, variable_speed=variable)


def rows():
    n = 12
    margin = np.linspace(1., .2, n).astype(np.float32)
    return dict(x=np.column_stack([margin, margin**2]), margin=margin,
                rate=np.ones(n, dtype=np.float32)*.1, y=margin*10,
                units=np.repeat([1, 2], 6), dataset=np.zeros(n, dtype=int),
                cycles=np.tile(np.arange(6)*10., 2), time=np.tile(np.arange(6)*.1, 2))


def test_initial_constant_rate_and_exact_boundary():
    net = model()
    x = torch.zeros(3, 2)
    m = torch.tensor([0., 1., 2.])
    r = torch.tensor([.5, .5, 2.])
    torch.testing.assert_close(net(x, m, r), m/r)
    torch.testing.assert_close(net.travel_time(x, m, r, m), torch.zeros(3))


def test_analytic_exponential_speed_integral():
    net = model()
    b = .7
    with torch.no_grad():
        net.net[-1].bias[1] = np.arctanh(b/2)
    m, r = torch.tensor([2.]), torch.tensor([.5])
    pred = net(torch.zeros(1, 2), m, r)
    torch.testing.assert_close(pred, m/r * np.expm1(b)/b)


def test_monotone_partial_time_and_gradient():
    net = model()
    with torch.no_grad():
        net.net[-1].bias[:] = torch.tensor([.3, -.4, .5])
    x = torch.ones(3, 2)
    m, r = torch.ones(3), torch.ones(3)*.2
    p = net.travel_time(x, m, r, torch.tensor([.8, .4, 0.]))
    assert torch.all(torch.diff(p) > 0)
    p.sum().backward()
    assert torch.isfinite(net.net[-1].bias.grad).all()
    assert torch.all(net.net[-1].bias.grad.abs() > 0)


def test_constant_ablation_ignores_shape_coefficients():
    net = model(False)
    with torch.no_grad():
        net.net[-1].bias[1:] = 3.
    torch.testing.assert_close(net(torch.zeros(1, 2), torch.ones(1), torch.ones(1)), torch.ones(1))


@pytest.mark.parametrize('margin,rate,target', [(-1., 1., 0.), (1., 0., 0.),
                                               (1., 1., 2.), (1., float('nan'), 0.)])
def test_invalid_coordinates_rejected(margin, rate, target):
    with pytest.raises(ValueError):
        model().travel_time(torch.zeros(1, 2), torch.tensor([margin]),
                            torch.tensor([rate]), torch.tensor([target]))


def test_pairs_causal_group_safe_and_label_independent():
    data = rows()
    pairs = make_prefix_pairs(data)
    a, b = pairs['anchor'], pairs['future']
    assert len(a) > 0
    assert np.all(data['units'][a] == data['units'][b])
    assert np.all(data['cycles'][b] > data['cycles'][a])
    assert np.all(data['margin'][b] < data['margin'][a])
    altered = {k: v.copy() for k, v in data.items()}
    altered['y'][:] = np.nan
    for k, v in pairs.items():
        np.testing.assert_array_equal(make_prefix_pairs(altered)[k], v)


def test_pair_order_invariant_and_duplicate_times_rejected():
    data = rows()
    p = make_prefix_pairs(data)
    shuffled = {k: v[::-1].copy() for k, v in data.items()}
    s = make_prefix_pairs(shuffled)
    np.testing.assert_allclose(p['elapsed'], s['elapsed'])
    np.testing.assert_allclose(p['target_margin'], s['target_margin'])
    data['time'][1] = data['time'][0]
    with pytest.raises(ValueError):
        make_prefix_pairs(data)


def test_batch_independence():
    net = model()
    with torch.no_grad():
        net.net[-1].weight.fill_(.1)
    x = torch.randn(4, 2)
    m, r = torch.ones(4), torch.ones(4)
    torch.testing.assert_close(net(x, m, r)[:1], net(x[:1], m[:1], r[:1]))


def test_training_reproducibility_and_fixed_epoch_refit():
    data = rows()
    first = fit_prefix_speed(data, data, seed=7, max_epochs=3, width=8, restore_best=False)
    second = fit_prefix_speed(data, data, seed=7, max_epochs=3, width=8, restore_best=False)
    np.testing.assert_array_equal(predict_prefix_speed(first, data), predict_prefix_speed(second, data))
    assert first.selection['executed_epochs'] == 3
    assert first.selection['source_pairs'] > 0


def test_more_quadrature_nodes_agree_at_coefficient_extremes():
    low = model()
    high = PrefixSpeedIntegral(np.zeros(2), np.ones(2), width=8, quadrature=64)
    with torch.no_grad():
        for net in (low, high):
            net.net[-1].bias[:] = torch.tensor([5., 5., 5.])
    args = (torch.zeros(2, 2), torch.ones(2), torch.ones(2), torch.tensor([0., .5]))
    torch.testing.assert_close(low.travel_time(*args), high.travel_time(*args), atol=1e-5, rtol=1e-5)
