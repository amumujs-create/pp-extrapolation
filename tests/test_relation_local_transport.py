import numpy as np
import pytest
import torch
from pp_extrapolation.relation_local_transport import (
    LocalRelationOperator, balanced_bank, fit_relation_transport,
    predict_relation_transport, source_episodes, validation_acceptance,
)


def data():
    grid = np.tile(np.linspace(.1, 1., 20), 4)
    groups = np.repeat(np.arange(4), 20).astype(str)
    x = np.column_stack((grid, np.sin(grid)))
    train = dict(x=x, y=20+30*grid, groups=groups)
    val = dict(x=x[:10]*1.1, y=20+33*grid[:10], groups=np.repeat('v', 10))
    return train, val


def test_constant_and_value_linearity():
    net = LocalRelationOperator(2)
    bank = torch.randn(20, 2, dtype=torch.float64)
    q = torch.randn(4, 2, dtype=torch.float64)
    w = torch.ones(20, dtype=torch.float64)
    u, v = torch.randn(2, 20, dtype=torch.float64)
    torch.testing.assert_close(net(q, bank, w*3, w), torch.ones(4, dtype=torch.float64)*3)
    torch.testing.assert_close(net(q, bank, 2*u-v, w), 2*net(q, bank, u, w)-net(q, bank, v, w))


def test_linear_extension_outside_training_range():
    net = LocalRelationOperator(1)
    with torch.no_grad():
        net.log_ridge.fill_(-9.)
        net.log_bandwidth.fill_(3.)
    x = torch.linspace(-1, 1, 20, dtype=torch.float64)[:, None]
    q = torch.tensor([[2.]], dtype=torch.float64)
    result = net(q, x, 3*x[:, 0]+10, torch.ones(20, dtype=torch.float64))
    assert abs(result.item()-16.) < .01


def test_gradients_and_batch_invariance():
    net = LocalRelationOperator(2)
    bank = torch.randn(25, 2, dtype=torch.float64)
    q = torch.randn(4, 2, dtype=torch.float64)
    y = bank[:, 0].square()+bank[:, 1]
    w = torch.ones(25, dtype=torch.float64)
    p = net(q, bank, y, w)
    torch.testing.assert_close(p[:1], net(q[:1], bank, y, w))
    p.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_episodes_exclude_units_and_support():
    train, _ = data()
    bank = balanced_bank(train['groups'], 40)
    eps = source_episodes(train['x'], train['groups'], bank, progress_index=0, increasing=True)
    for e in eps:
        a, q = e['anchor'], e['query']
        assert not set(train['groups'][a]) & set(train['groups'][q])
        assert train['x'][a, 0].max() < train['x'][q, 0].min()


def test_prior_rejection_ignores_unavailable_prior_and_boundary():
    train, val = data()
    fit = fit_relation_transport(train, val, progress_index=0, increasing=True,
        prior_approved=False, prior_train='unusable', prior_validation='unusable', boundary_index=999,
        max_epochs=5)
    p = predict_relation_transport(fit, val['x'], prior_prediction='unusable')
    assert np.isfinite(p).all() and not fit.prior_approved
    assert fit.boundary_index is None


def test_exact_prior_zero_residual_and_boundary():
    train, val = data()
    fit = fit_relation_transport(train, val, progress_index=0, increasing=True,
        prior_approved=True, prior_train=train['y'], prior_validation=val['y'], boundary_index=0,
        learn=False)
    np.testing.assert_allclose(predict_relation_transport(fit, val['x'], prior_prediction=val['y']), val['y'])
    np.testing.assert_array_equal(predict_relation_transport(fit, np.zeros((1, 2)), prior_prediction=np.zeros(1)), [0.])


def test_level_only_and_full_differ_on_slope_error():
    train, val = data()
    fit = fit_relation_transport(train, val, progress_index=0, increasing=True,
        prior_approved=True, prior_train=train['y']-10*train['x'][:, 0],
        prior_validation=val['y']-10*val['x'][:, 0], learn=False)
    q = np.array([[2., np.sin(2.)]])
    full = predict_relation_transport(fit, q, prior_prediction=np.array([60.]))
    level = predict_relation_transport(fit, q, prior_prediction=np.array([60.]), level_only=True)
    assert abs(full[0]-level[0]) > 1.


def test_input_and_split_rejections():
    train, val = data()
    with pytest.raises(ValueError):
        fit_relation_transport(train, train, progress_index=0, increasing=True)
    with pytest.raises(ValueError):
        fit_relation_transport(train, val, progress_index=0, increasing=True, prior_approved=True)
    with pytest.raises(ValueError):
        balanced_bank(train['groups'], 1)


def test_seed_reproducibility():
    train, val = data()
    fits = [fit_relation_transport(train, val, progress_index=0, increasing=True, seed=42, max_epochs=5) for _ in range(2)]
    np.testing.assert_array_equal(predict_relation_transport(fits[0], val['x']), predict_relation_transport(fits[1], val['x']))


def test_validation_gate_requires_unit_safety_and_exact_tie_rejects():
    y = np.zeros(4)
    g = np.array([1, 1, 2, 2])
    b = np.ones(4)
    assert validation_acceptance(y, g, b, b*.5)['accepted']
    assert not validation_acceptance(y, g, b, b)['accepted']
    assert not validation_acceptance(y, g, b, np.array([0., 0., 1.1, 1.1]))['accepted']
