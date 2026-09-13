import pytest
import torch

from pp_extrapolation.profiled_coordinate_flow import ProfiledCoordinateFlow


def inputs():
    t = torch.arange(6, dtype=torch.float32)[None].repeat(2, 1)/32
    x = torch.stack((2-t[0], 1+.5*t[0]))[..., None]
    return x, t, torch.ones_like(t, dtype=torch.bool), torch.tensor([[0., 1.], [0., 2.]])


def test_exact_boundary_and_coordinate_translation():
    model = ProfiledCoordinateFlow()
    x, t, mask, h = inputs()
    p = model.forecast(x, t, mask, h)
    torch.testing.assert_close(p[:, 0], x[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(p, model.forecast(x+5, t+10, mask, h)-5)


def test_padding_cannot_change_prediction_or_poison_solve():
    model = ProfiledCoordinateFlow()
    x, t, mask, h = inputs()
    p = model.forecast(x, t, mask, h)
    xp = torch.cat((x, torch.full((2, 3, 1), float('nan'))), 1)
    tp = torch.cat((t, torch.full((2, 3), float('nan'))), 1)
    mp = torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), 1)
    torch.testing.assert_close(p, model.forecast(xp, tp, mp, h))


def test_batch_independence_and_gradients_through_identification():
    model = ProfiledCoordinateFlow()
    x, t, mask, h = inputs()
    p = model.forecast(x, t, mask, h)
    torch.testing.assert_close(p[:1], model.forecast(x[:1], t[:1], mask[:1], h[:1]))
    p.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.log_tau.grad.abs().sum() > 0


def test_invalid_coordinate_and_mask_fail_explicitly():
    model = ProfiledCoordinateFlow()
    x, t, mask, h = inputs()
    mask[0, 1] = False
    with pytest.raises(ValueError):
        model.forecast(x, t, mask, h)
    with pytest.raises(ValueError):
        model.forecast(x, -t, torch.ones_like(mask), h)
