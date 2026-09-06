import numpy as np
import pytest
import torch
from pp_extrapolation import CounterfactualRays, PriorPairs, TransportTriples, fit_pp, predict
from pp_extrapolation.priors import (counterfactual_residual_loss, pair_loss,
    prepare_counterfactual, prepare_pairs, prepare_transport, transport_loss)

class Linear(torch.nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(float(slope)))
    def forward(self, x):
        return self.slope*x[:,0]

def test_degradation_prior_detects_wrong_sign_and_corrects_gradient():
    p = PriorPairs(np.ones((3,1)), np.zeros((3,1)), np.full(3,-np.inf), np.zeros(3), np.ones(3))
    v = prepare_pairs(p,np.zeros(1),np.ones(1),1.)
    assert pair_loss(Linear(1),v).item() == 0
    wrong = Linear(-1)
    loss = pair_loss(wrong,v)
    loss.backward()
    assert loss.item() == 1
    assert wrong.slope.grad.item() < 0
    p.confidence[:] = 0
    assert pair_loss(wrong,prepare_pairs(p,np.zeros(1),np.ones(1),1.)).item() == 0

def test_transport_respects_unequal_path_steps():
    p = TransportTriples(np.array([[2.]]),np.array([[1.]]),np.array([[-1.]]),np.array([2.]),np.zeros(1),np.ones(1))
    v = prepare_transport(p,np.zeros(1),np.ones(1),1.)
    assert transport_loss(Linear(3),v).item() == 0
    class Square(torch.nn.Module):
        def forward(self,x): return x[:,0]**2
    assert transport_loss(Square(),v).item() == 36

def test_zero_weight_is_exact_baseline_and_enabled_fit_runs():
    x = np.linspace(.5,1,20)[:,None].astype('float32')
    tr = dict(x=x,y=(10*x[:,0]).astype('float32'),groups=np.repeat(['a','b'],10))
    va = dict(x=x[:5]-.4,y=(10*(x[:5,0]-.4)).astype('float32'),groups=np.repeat('v',5))
    p = PriorPairs(x,x-.1,np.full(20,-np.inf),np.zeros(20),np.ones(20))
    base = fit_pp(tr,va,seed=42,max_epochs=3)
    off = fit_pp(tr,va,seed=42,max_epochs=3,prior_pairs=p,prior_weight=0)
    np.testing.assert_array_equal(predict(base,va['x']),predict(off,va['x']))
    enabled = fit_pp(tr,va,seed=42,max_epochs=3,prior_pairs=p,prior_weight=1)
    assert np.isfinite(predict(enabled,va['x'])).all()
    with pytest.raises(ValueError): fit_pp(tr,va,seed=42,prior_weight=1)

def test_counterfactual_residual_contract_penalizes_wrong_decay():
    model = type("ResidualModel", (torch.nn.Module,), {})()
    model.nonlinear = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.nonlinear.weight.fill_(1.)
    p = CounterfactualRays(np.array([[1.]]), np.array([[2.]]), np.array([.5]), np.ones(1))
    values = prepare_counterfactual(p, np.zeros(1), np.ones(1))
    assert counterfactual_residual_loss(model, values).item() == pytest.approx(2.25)

def test_counterfactual_fit_requires_rays_when_enabled():
    x=np.linspace(0,1,10)[:,None].astype('float32')
    split=dict(x=x,y=x[:,0],groups=np.repeat(['a','b'],5))
    with pytest.raises(ValueError): fit_pp(split,split,seed=1,counterfactual_weight=.1)
