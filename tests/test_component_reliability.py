import numpy as np
import torch
from pp_extrapolation.component_reliability import Reliability,predict_reliability


def prepared():
    return dict(mean=torch.ones((5,3),dtype=torch.float64),cov=torch.eye(3,dtype=torch.float64)*.2,
        prior_mean=torch.zeros(3,dtype=torch.float64),prior_cov=torch.eye(3,dtype=torch.float64),
        features=torch.randn((5,3,4),generator=torch.Generator().manual_seed(4),dtype=torch.float64),
        margin=torch.tensor([0,.1,.2,.3,.4],dtype=torch.float64))


def test_ridge_reproduces_coefficient_distribution():
    p=prepared();mu,cov,w=Reliability('ridge').distribution(p)
    torch.testing.assert_close(mu,p['mean']);torch.testing.assert_close(cov,p['cov'].expand(5,-1,-1))


def test_tied_gate_identical_across_components():
    _,_,w=Reliability('tied').distribution(prepared())
    torch.testing.assert_close(w[:,0],w[:,1]);torch.testing.assert_close(w[:,1],w[:,2])


def test_distribution_psd_bounded_gate_and_boundary():
    p=prepared();model=Reliability();_,cov,w=model.distribution(p)
    assert (torch.linalg.eigvalsh(cov)>0).all() and ((w>0)&(w<1)).all()
    draws,_=predict_reliability(model,p,32)
    np.testing.assert_array_equal(draws[0],0);assert (draws[1:]>0).all()
    np.testing.assert_array_equal(draws,predict_reliability(model,p,32)[0])


def test_gate_receives_gradients():
    model=Reliability();model.draws(prepared(),torch.zeros((4,3),dtype=torch.float64)).sum().backward()
    assert torch.isfinite(model.weight.grad).all() and model.weight.grad.abs().sum()>0
