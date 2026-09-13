import numpy as np
import pytest
import torch
from pp_extrapolation.component_transfer import (
    make_bank, component_features, ComponentFit, ComponentGate,
    fit_components, predict_components,
)
from pp_extrapolation.relation_local_transport import LocalRelationOperator


def rows():
    t = np.tile(np.linspace(.1, 1., 15), 4)
    return dict(x=np.column_stack((t, np.sin(t))), y=5+10*t,
                groups=np.repeat(['a', 'b', 'c', 'd'], 15))


def test_components_reconstruct_local_operator():
    r = rows(); base = r['y']*.8
    bank = make_bank(r, base)
    query = np.array([[1.5, 1.], [.5, .4]])
    c, z = component_features(bank, query, np.ones(2))
    net = LocalRelationOperator(2)
    with torch.no_grad():
        p = net(torch.tensor((query-bank.center)/bank.scale), torch.tensor(bank.x),
                torch.tensor(bank.values), torch.tensor(bank.weights)).numpy()*bank.value_scale
    np.testing.assert_allclose(c.sum(1), p, rtol=1e-10, atol=1e-10)
    assert z.shape == (2, 8)


def test_exact_prior_has_no_correction_and_boundary_is_exact():
    r = rows(); bank = make_bank(r, r['y'], boundary_index=0)
    c, _ = component_features(bank, r['x'], r['y'])
    np.testing.assert_array_equal(c, 0.)
    bank = make_bank(r, .8*r['y'], boundary_index=0)
    c, _ = component_features(bank, np.zeros((2, 2)), np.zeros(2))
    np.testing.assert_array_equal(c, 0.)
    with pytest.raises(ValueError):
        component_features(bank, np.zeros((2, 2)), np.ones(2))


def test_batch_invariance_and_no_outcome_input():
    r = rows(); bank = make_bank(r, .8*r['y'])
    q = r['x'][:4]; base = r['y'][:4]*.8
    c, z = component_features(bank, q, base)
    c1, z1 = component_features(bank, q[:1], base[:1])
    np.testing.assert_allclose(c[:1], c1)
    np.testing.assert_allclose(z[:1], z1)


def test_disabled_gate_preserves_base():
    model = ComponentGate()
    fit = ComponentFit(model, np.zeros(8), np.ones(8), False, None, {})
    p, g = predict_components(fit, np.ones((3, 3)), np.zeros((3, 8)), np.arange(3.), return_gates=True)
    np.testing.assert_array_equal(p, np.arange(3.))
    np.testing.assert_array_equal(g, 0.)


def test_gates_bounded_and_differentiable():
    for conditional in (True, False):
        net = ComponentGate(conditional)
        g = net(torch.randn(4, 8))
        assert torch.all((g >= 0) & (g <= 1))
        g.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_crossfit_correction_learning_and_reproducibility():
    n = 30
    episode = dict(c=np.column_stack((np.ones(n)*2, np.zeros((n, 2)))), z=np.zeros((n, 8)),
                   base=np.ones(n)*10, y=np.ones(n)*12, groups=np.repeat(['a','b','c'], 10))
    val = {**episode, 'groups': np.repeat('v', n)}
    fits = [fit_components([episode], val, seed=42, conditional=False, max_epochs=20) for _ in range(2)]
    p = [predict_components(f, val['c'], val['z'], val['base']) for f in fits]
    assert fits[0].enabled
    assert np.mean((p[0]-12)**2) < 4
    np.testing.assert_array_equal(p[0], p[1])


def test_validation_can_disable_harmful_transfer():
    n = 12
    episode = dict(c=np.ones((n, 3)), z=np.zeros((n, 8)), base=np.ones(n)*10,
                   y=np.ones(n)*12, groups=np.repeat('a', n))
    val = {**episode, 'y':np.ones(n)*10, 'groups':np.repeat('v', n)}
    fit = fit_components([episode], val, max_epochs=10)
    assert not fit.enabled and fit.selection['selected_epoch'] == -1


def test_invalid_inputs():
    r=rows()
    with pytest.raises(ValueError):
        make_bank(r, np.zeros(1))
    with pytest.raises(ValueError):
        make_bank(r, r['y'], progress_index=999)
    with pytest.raises(ValueError):
        fit_components([], {})
