"""Prefix-to-future latent trajectory operator for strict temporal extrapolation.

This is intentionally not a PP-X residual model.  It learns one latent vector
field from source prefixes and obtains every future prediction only by
integrating that field beyond the last observed time.  A point-RUL head is not
part of the architecture: a threshold-crossing time, if wanted, is derived
after a trajectory has been extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class LatentTrajectoryOperator(nn.Module):
    """Causal filter + autonomous latent flow + observation decoder."""

    def __init__(self, input_dim: int, output_dim: int, *, width: int = 32,
                 latent_dim: int = 12):
        super().__init__()
        self.encoder = nn.GRU(input_dim + 1, width, batch_first=True)
        self.to_latent = nn.Sequential(nn.Linear(width, latent_dim), nn.Tanh())
        self.vector_field = nn.Sequential(
            nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, output_dim)
        )
        # Starts as a persistence flow.  Training must earn nonzero dynamics.
        nn.init.zeros_(self.vector_field[-1].weight)
        nn.init.zeros_(self.vector_field[-1].bias)

    def encode(self, x: torch.Tensor, time: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or time.shape != x.shape[:2] or mask.shape != x.shape[:2]:
            raise ValueError("x, time, mask must align on (batch, step)")
        elapsed = time - time[:, :1]
        hidden, _ = self.encoder(torch.cat((x, elapsed[..., None]), dim=-1))
        last = mask.long().sum(1).clamp_min(1) - 1
        return self.to_latent(hidden[torch.arange(len(x), device=x.device), last])

    def advance(self, latent: torch.Tensor, horizon: torch.Tensor,
                *, steps: int = 16) -> torch.Tensor:
        """Fixed-step differentiable integration for one horizon per row."""
        if horizon.ndim != 1 or len(horizon) != len(latent) or torch.any(horizon < 0):
            raise ValueError("horizon must be a nonnegative vector aligned to latent")
        state = latent
        dt = horizon[:, None] / int(steps)
        for _ in range(int(steps)):
            state = state + dt * self.vector_field(state)
        return state

    def forecast(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor,
                 horizons: torch.Tensor, *, steps: int = 16) -> torch.Tensor:
        """Forecast at arbitrary future horizons, never through a direct head."""
        if horizons.ndim != 2 or horizons.shape[0] != len(x):
            raise ValueError("horizons must have shape (batch, future_step)")
        latent = self.encode(x, time, mask)
        output = [self.decoder(self.advance(latent, horizons[:, j], steps=steps))
                  for j in range(horizons.shape[1])]
        return torch.stack(output, dim=1)


def semigroup_error(model: LatentTrajectoryOperator, latent: torch.Tensor,
                    first_horizon: torch.Tensor, second_horizon: torch.Tensor,
                    *, steps: int = 16) -> torch.Tensor:
    """Measure flow composition error, a diagnostic specific to extrapolation.

    `Phi(t1+t2,z)` should agree with `Phi(t2,Phi(t1,z))`.  It is zero only for
    an exactly solved flow, not automatically for a discretized neural rollout.
    """
    once = model.advance(latent, first_horizon + second_horizon, steps=steps)
    twice = model.advance(model.advance(latent, first_horizon, steps=steps),
                          second_horizon, steps=steps)
    return (once - twice).square().mean(1)


@dataclass(frozen=True)
class MultiHorizonBatch:
    """A source-only supervision batch made from a trajectory's own suffix."""
    x: np.ndarray
    time: np.ndarray
    mask: np.ndarray
    horizons: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray

    def validate(self) -> None:
        n = len(self.x)
        if (self.x.ndim != 3 or self.time.shape != self.x.shape[:2]
                or self.mask.shape != self.x.shape[:2] or self.horizons.ndim != 2
                or self.horizons.shape[0] != n or self.target.ndim != 3
                or self.target.shape[:2] != self.horizons.shape
                or self.target_mask.shape != self.horizons.shape
                or np.any(self.horizons < 0)):
            raise ValueError("invalid multi-horizon extrapolation batch")


def multi_horizon_loss(model: LatentTrajectoryOperator, batch: MultiHorizonBatch,
                       *, steps: int = 16) -> torch.Tensor:
    """Masked suffix-rollout loss; targets must be future of the input prefix."""
    batch.validate()
    prediction = model.forecast(torch.as_tensor(batch.x, dtype=torch.float32),
                                 torch.as_tensor(batch.time, dtype=torch.float32),
                                 torch.as_tensor(batch.mask, dtype=torch.bool),
                                 torch.as_tensor(batch.horizons, dtype=torch.float32),
                                 steps=steps)
    truth = torch.as_tensor(batch.target, dtype=torch.float32)
    valid = torch.as_tensor(batch.target_mask, dtype=torch.bool)[..., None]
    if not torch.any(valid):
        raise ValueError("at least one future target is required")
    return ((prediction - truth).square() * valid).sum() / (valid.sum() * truth.shape[-1])
