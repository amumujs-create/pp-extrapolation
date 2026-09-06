import numpy as np
import torch
from pp_extrapolation.regime_spline import RegimeSplineFit,RegimeSplineNet,jacobian_diagnostics

def test_monotone_hinge_component_does_not_increase():
    model=RegimeSplineNet(1,np.array([0.,.5],dtype=np.float32),1.,monotone=True)
    with torch.no_grad():
        model.affine.weight.zero_();model.affine.bias.zero_()
        for p in model.local.parameters():p.zero_()
    y=model(torch.tensor([[0.],[.5],[1.],[2.]])).detach().numpy()
    assert np.all(np.diff(y)<=1e-7)

def test_jacobian_diagnostics_uses_oriented_degradation():
    model=RegimeSplineNet(1,np.array([0.],dtype=np.float32),-1.)
    with torch.no_grad():
        model.affine.weight.fill_(1.);model.affine.bias.zero_()
        for p in model.context.parameters():p.zero_()
        for p in model.local.parameters():p.zero_()
    fit=RegimeSplineFit(model,np.zeros(1,dtype=np.float32),np.ones(1,dtype=np.float32),1.,{'direction':-1.})
    diagnostic=jacobian_diagnostics(fit,np.array([[0.],[1.]],dtype=np.float32))
    assert diagnostic['violation_fraction']==0.
