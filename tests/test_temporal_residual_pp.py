import numpy as np
import torch

from pp_extrapolation.temporal_residual_pp import TemporalResidualPP


def test_causal_prefix_and_padding_mask_invariance():
    torch.manual_seed(1)
    model = TemporalResidualPP(3, 8, mode="pp", residual_bound=.5,
                               support_decay=.1, dropout=0.)
    with torch.no_grad():
        model.support_min.fill_(-2)
        model.support_max.fill_(2)
    prefix = torch.randn(1, 8, 3)
    left = torch.zeros(1, 4, 3)
    padded = torch.cat([left, prefix], dim=1)
    mask = torch.cat([torch.zeros(1, 4, dtype=torch.bool),
                      torch.ones(1, 8, dtype=torch.bool)], dim=1)
    plain = model(prefix, torch.ones(1, 8, dtype=torch.bool))
    extended = model(padded, mask)
    torch.testing.assert_close(plain, extended, atol=1e-5, rtol=1e-5)


def test_uncapped_output_is_finite_beyond_training_support():
    model = TemporalResidualPP(2, 4, mode="pp", residual_bound=1.,
                               support_decay=.5, dropout=0.)
    with torch.no_grad():
        model.support_min.fill_(-1)
        model.support_max.fill_(1)
        model.affine.weight.fill_(2)
        model.affine.bias.zero_()
    x = torch.full((3, 10, 2), 5.)
    prediction = model(x, torch.ones(3, 10, dtype=torch.bool))
    assert torch.isfinite(prediction).all()
    assert (prediction > 1).all()
