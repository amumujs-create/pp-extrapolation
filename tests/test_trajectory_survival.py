import numpy as np
import torch

from pp_extrapolation.trajectory_survival import TrajectorySurvivalNet


def test_survival_is_bounded_and_monotone_in_future_time():
    net = TrajectorySurvivalNet(3, width=6, latent_dim=3)
    x = torch.randn(4, 5, 3)
    time = torch.arange(5, dtype=torch.float32).repeat(4, 1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    curve = net.survival(x, time, mask, torch.tensor([0.0, 2.0, 4.0]))
    assert torch.allclose(curve[:, 0], torch.ones(4))
    assert torch.all(curve[:, 1:] <= curve[:, :-1])
    assert torch.all((curve >= 0) & (curve <= 1))


def test_future_damage_is_monotone():
    net = TrajectorySurvivalNet(2, width=5, latent_dim=2)
    x = torch.randn(3, 4, 2)
    time = torch.arange(4, dtype=torch.float32).repeat(3, 1)
    mask = torch.ones(3, 4, dtype=torch.bool)
    context, damage = net.filter(x, time, mask)
    velocity, _ = net._rate_and_hazard(context, damage)
    assert np.all((damage + velocity * 3).detach().numpy() >= damage.detach().numpy())
