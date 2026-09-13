import numpy as np
import torch

from pp_extrapolation.trajectory_operator import (
    LatentTrajectoryOperator,
    MultiHorizonBatch,
    multi_horizon_loss,
    semigroup_error,
)


def test_zero_initialized_flow_is_exact_persistence_extrapolator():
    net = LatentTrajectoryOperator(2, 1, width=5, latent_dim=3)
    x = torch.randn(3, 4, 2)
    time = torch.arange(4, dtype=torch.float32).repeat(3, 1)
    mask = torch.ones(3, 4, dtype=torch.bool)
    forecast = net.forecast(x, time, mask, torch.tensor([[1., 5.]] * 3))
    assert torch.allclose(forecast[:, 0], forecast[:, 1])


def test_multi_horizon_loss_only_accepts_future_aligned_targets():
    batch = MultiHorizonBatch(
        x=np.zeros((2, 3, 2), np.float32), time=np.tile(np.arange(3), (2, 1)).astype(np.float32),
        mask=np.ones((2, 3), bool), horizons=np.array([[1., 2.], [1., 2.]], np.float32),
        target=np.zeros((2, 2, 1), np.float32), target_mask=np.ones((2, 2), bool),
    )
    assert multi_horizon_loss(LatentTrajectoryOperator(2, 1, width=4, latent_dim=2), batch).item() >= 0


def test_semigroup_error_is_zero_for_initial_persistence_flow():
    net = LatentTrajectoryOperator(2, 1, width=5, latent_dim=3)
    error = semigroup_error(net, torch.randn(4, 3), torch.ones(4), torch.full((4,), 2.))
    assert torch.allclose(error, torch.zeros(4))
