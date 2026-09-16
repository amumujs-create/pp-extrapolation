import numpy as np
import torch

from pp_extrapolation.regime_mixture import LatentRegimePPNet


def test_affine_scale_zero_removes_only_affine_path():
    torch.manual_seed(7)
    prior = LatentRegimePPNet(3, 1.0, 0.0, width=4, affine_scale=1.0)
    torch.manual_seed(7)
    direct = LatentRegimePPNet(3, 1.0, 0.0, width=4, affine_scale=0.0)
    x = torch.tensor(np.arange(12).reshape(4, 3) / 10, dtype=torch.float32)
    with torch.no_grad():
        prior_pred, prior_gate, prior_tails = prior.components(x)
        direct_pred, direct_gate, direct_tails = direct.components(x)
        affine = prior.affine(x)
    assert torch.allclose(prior_gate, direct_gate)
    assert all(torch.allclose(a, b) for a, b in zip(prior_tails, direct_tails))
    assert torch.allclose(prior_pred - direct_pred, affine)
