import numpy as np
import torch
from pp_extrapolation.regime_spline import RegimeSplineNet

def test_monotone_hinge_component_does_not_increase():
    model=RegimeSplineNet(1,np.array([0.,.5],dtype=np.float32),1.,monotone=True)
    with torch.no_grad():
        model.affine.weight.zero_();model.affine.bias.zero_()
        for p in model.local.parameters():p.zero_()
    y=model(torch.tensor([[0.],[.5],[1.],[2.]])).detach().numpy()
    assert np.all(np.diff(y)<=1e-7)
