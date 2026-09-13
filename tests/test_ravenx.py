import numpy as np
import torch
from pp_extrapolation.piecewise_health_path import one_switch_rul,no_switch_rul
from pp_extrapolation.relation_intervention_generator import RelationInterventionGenerator,stretched_gate
from pp_extrapolation.regime_validity_attention import sparsemax,PreparedWindows,RegimeValidityAttention,mixture_energy
from pp_extrapolation.sticky_regime_validity_attention import StickyRegimeValidityAttention


def prepared(n=5):
    gen=torch.Generator().manual_seed(3)
    return PreparedWindows(torch.randn((n,4,15),generator=gen,dtype=torch.float64),
        torch.randn((n,4,3),generator=gen,dtype=torch.float64)*.1,
        torch.tensor([[False,True,True,True]]*n),torch.linspace(0,1,n,dtype=torch.float64),
        torch.linspace(0,.5,n,dtype=torch.float64),torch.eye(3,dtype=torch.float64)*.1,
        torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64))


def test_sparsemax_is_sparse_probability():
    p=sparsemax(torch.tensor([[10.,0.,-2.]],dtype=torch.float64))
    torch.testing.assert_close(p.sum(1),torch.ones(1,dtype=torch.float64));assert p[0,1]==p[0,2]==0


def test_piecewise_identity_and_boundary():
    theta=torch.randn((4,7,3),generator=torch.Generator().manual_seed(2),dtype=torch.float64)*.1
    margin=torch.linspace(0,1,4,dtype=torch.float64)
    for switch in (torch.zeros(4,dtype=torch.float64),margin):
        torch.testing.assert_close(one_switch_rul(theta,theta,margin,switch),no_switch_rul(theta,margin))
    np.testing.assert_array_equal(one_switch_rul(theta,theta,margin,margin)[0].numpy(),0.)


def test_relation_mask_blocks_level_curvature():
    model=RelationInterventionGenerator();features=torch.ones((2,2,15),dtype=torch.float64)
    location,_,_=model.parameters_for(features)
    assert torch.all(location[:,0,2]==0)
    assert ((stretched_gate(torch.tensor([-100.,100.]))==torch.tensor([0.,1.])).all())


def test_mixture_probabilities_and_gradients():
    model=RegimeValidityAttention();p=prepared();noise=torch.zeros((4,3),dtype=torch.float64)
    draws,prob,attn,_=model(p,noise)
    assert draws.shape==(5,17,4);torch.testing.assert_close(prob.sum(1),torch.ones(5,dtype=torch.float64))
    torch.testing.assert_close(attn.sum(2),torch.ones((5,3),dtype=torch.float64))
    loss=mixture_energy(draws,prob,torch.ones(5,dtype=torch.float64)).mean();loss.backward()
    assert any(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())


def test_no_transition_is_exact_atom():
    model=RegimeValidityAttention(attention='softmax',transitions=False);p=prepared(2)
    _,prob,_,_=model(p,torch.zeros((3,3),dtype=torch.float64))
    np.testing.assert_array_equal(prob[:,0].detach().numpy(),1);np.testing.assert_array_equal(prob[:,1:].detach().numpy(),0)


def test_sticky_initialization_retains_no_switch_mass():
    model=StickyRegimeValidityAttention();p=prepared(3)
    with torch.no_grad():
        h,a=model.window_attention(p);context=torch.einsum('bkw,bwh->bkh',a,h).mean(1)
        prob=model.path_probabilities(context,p.distance)
    assert torch.all(prob[:,0]>.3)
