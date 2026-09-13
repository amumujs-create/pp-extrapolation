"""Causal latent-damage survival model, independent of PP-X.

The model predicts a failure-time distribution from a causal sensor prefix.  It
has no affine extrapolator, hand-written prior, support-distance gate, or
validation-selected route.  A positive damage velocity makes the health state
monotone in forecast time; a positive hazard turns that state into a survival
curve.  This is deliberately an experimental research module, not a PP-X
executor.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import equal_group_weights


class TrajectorySurvivalNet(nn.Module):
    """Filter a sensor prefix, then propagate a monotone latent damage state."""

    def __init__(self, feature_dim: int, width: int = 24, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.GRU(feature_dim + 1, width, batch_first=True)
        self.context = nn.Sequential(nn.Linear(width, latent_dim), nn.Tanh())
        self.damage_head = nn.Linear(width, 1)
        self.velocity = nn.Sequential(
            nn.Linear(latent_dim + 1, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.hazard = nn.Sequential(
            nn.Linear(latent_dim + 1, width), nn.SiLU(), nn.Linear(width, 1)
        )
        # A small positive initial hazard is less pathological than an
        # uncalibrated direct RUL head, and all output constraints are exact.
        nn.init.zeros_(self.damage_head.weight)
        nn.init.zeros_(self.damage_head.bias)
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.constant_(self.velocity[-1].bias, -2.0)
        nn.init.zeros_(self.hazard[-1].weight)
        nn.init.constant_(self.hazard[-1].bias, -2.0)

    def filter(self, x: torch.Tensor, time: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context and damage using no samples after each prefix end."""
        if x.ndim != 3 or time.shape != x.shape[:2] or mask.shape != x.shape[:2]:
            raise ValueError("x, time, and mask must have aligned (batch, step) axes")
        elapsed = time - time[:, :1]
        sequence = torch.cat((x, elapsed[..., None]), dim=-1)
        encoded, _ = self.encoder(sequence)
        lengths = mask.long().sum(1).clamp_min(1) - 1
        final = encoded[torch.arange(len(x), device=x.device), lengths]
        context = self.context(final)
        damage = torch.nn.functional.softplus(self.damage_head(final)).squeeze(1)
        return context, damage

    def _rate_and_hazard(self, context: torch.Tensor,
                         damage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat((context, damage[:, None]), dim=1)
        velocity = torch.nn.functional.softplus(self.velocity(value).squeeze(1)) + 1e-6
        hazard = torch.nn.functional.softplus(self.hazard(value).squeeze(1)) + 1e-6
        return velocity, hazard

    def cumulative_hazard(self, x: torch.Tensor, time: torch.Tensor,
                          mask: torch.Tensor, horizon: torch.Tensor,
                          steps: int = 32) -> torch.Tensor:
        """Euler-integrate a future hazard for a per-row nonnegative horizon."""
        if horizon.ndim != 1 or len(horizon) != len(x) or torch.any(horizon < 0):
            raise ValueError("horizon must be one nonnegative value per row")
        context, damage = self.filter(x, time, mask)
        dt = horizon / int(steps)
        cumulative = torch.zeros_like(horizon)
        for _ in range(int(steps)):
            velocity, hazard = self._rate_and_hazard(context, damage)
            cumulative = cumulative + hazard * dt
            damage = damage + velocity * dt
        return cumulative

    def survival(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor,
                 horizons: torch.Tensor, steps: int = 32) -> torch.Tensor:
        """Survival curves; columns correspond exactly to ``horizons``."""
        if horizons.ndim != 1:
            raise ValueError("horizons must be one-dimensional")
        hazards = [self.cumulative_hazard(x, time, mask, h.expand(len(x)), steps)
                   for h in horizons]
        return torch.exp(-torch.stack(hazards, dim=1))


@dataclass
class TrajectorySurvivalFit:
    model: TrajectorySurvivalNet
    center: np.ndarray
    scale: np.ndarray
    horizon_cap: float
    selection: dict


def _validate(split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(split["x"], dtype=np.float32)
    time = np.asarray(split["time"], dtype=np.float32)
    mask = np.asarray(split["mask"], dtype=bool)
    rul = np.asarray(split["rul"], dtype=np.float32)
    event = np.asarray(split.get("event", np.ones(len(x), dtype=bool)), dtype=bool)
    groups = np.asarray(split["groups"])
    if (x.ndim != 3 or time.shape != x.shape[:2] or mask.shape != x.shape[:2]
            or rul.shape != (len(x),) or event.shape != (len(x),)
            or groups.shape != (len(x),) or not np.isfinite(x[mask]).all()
            or not np.isfinite(time[mask]).all() or np.any(rul < 0)):
        raise ValueError("invalid causal trajectory-survival split")
    if np.any(mask.sum(1) == 0):
        raise ValueError("each trajectory needs at least one observed step")
    return x, time, mask, rul, event, groups


def fit_trajectory_survival(train: dict, validation: dict, *, seed: int = 42,
                            width: int = 24, latent_dim: int = 8,
                            learning_rate: float = 5e-4, weight_decay: float = 1e-3,
                            max_epochs: int = 300, patience: int = 50,
                            integration_steps: int = 32) -> TrajectorySurvivalFit:
    """Fit censored event likelihood, selecting only by validation NLL.

    ``event=False`` is a right-censored row: ``rul`` then means its observed
    censoring horizon.  This permits unfinished units without inventing an RUL
    label.  Evaluation still requires a held-out failure-time protocol.
    """
    tx, tt, tm, tr, te, tg = _validate(train)
    vx, vt, vm, vr, ve, _ = _validate(validation)
    observed = tx[tm]
    center = observed.mean(0)
    scale = observed.std(0)
    scale[scale < 1e-6] = 1.0
    tx = ((tx - center) / scale).astype(np.float32)
    vx = ((vx - center) / scale).astype(np.float32)
    tx[~tm] = 0.0
    vx[~vm] = 0.0
    cap = max(float(np.quantile(tr, .98)), 1.0)
    torch.manual_seed(int(seed))
    model = TrajectorySurvivalNet(tx.shape[2], width, latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    x, t, m = map(torch.as_tensor, (tx, tt, tm))
    y, event = torch.as_tensor(tr), torch.as_tensor(te)
    weights = torch.as_tensor(equal_group_weights(tg), dtype=torch.float32)
    xv, tv, mv = map(torch.as_tensor, (vx, vt, vm))
    yv, ev = torch.as_tensor(vr), torch.as_tensor(ve)
    rng = np.random.default_rng(seed)

    def negative_log_likelihood(x0, t0, m0, y0, event0, weight=None):
        cumulative = model.cumulative_hazard(x0, t0, m0, y0, integration_steps)
        # log h(T) is recovered stably as the final tiny interval's rate.
        context, damage = model.filter(x0, t0, m0)
        horizon = y0 / int(integration_steps)
        for _ in range(int(integration_steps) - 1):
            velocity, _ = model._rate_and_hazard(context, damage)
            damage = damage + velocity * horizon
        _, hazard = model._rate_and_hazard(context, damage)
        loss = cumulative - event0.to(cumulative.dtype) * torch.log(hazard)
        return (loss if weight is None else loss * weight).mean()

    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tx))
        for start in range(0, len(tx), 256):
            index = torch.as_tensor(order[start:start + 256])
            loss = negative_log_likelihood(x[index], t[index], m[index], y[index],
                                           event[index], weights[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(negative_log_likelihood(xv, tv, mv, yv, ev))
        history.append({"epoch": epoch, "validation_nll": score})
        if score < best - 1e-8:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break
    model.load_state_dict(best_state)
    return TrajectorySurvivalFit(model, center, scale, cap, {
        "seed": seed, "width": width, "latent_dim": latent_dim,
        "selected_epoch": best_epoch, "validation_nll": best, "history": history,
    })


def predict_survival(fit: TrajectorySurvivalFit, split: dict, horizons: np.ndarray,
                     *, integration_steps: int = 32) -> np.ndarray:
    x, time, mask, _, _, _ = _validate(split)
    values = ((x - fit.center) / fit.scale).astype(np.float32)
    values[~mask] = 0.0
    grid = np.asarray(horizons, dtype=np.float32)
    if grid.ndim != 1 or np.any(grid < 0) or np.any(np.diff(grid) < 0):
        raise ValueError("horizons must be sorted nonnegative values")
    fit.model.eval()
    with torch.no_grad():
        return fit.model.survival(torch.as_tensor(values), torch.as_tensor(time),
                                  torch.as_tensor(mask), torch.as_tensor(grid),
                                  integration_steps).cpu().numpy()


def predict_mean_rul(fit: TrajectorySurvivalFit, split: dict, *, points: int = 64) -> np.ndarray:
    """Restricted mean survival time, never an unconstrained regression head."""
    grid = np.linspace(0.0, fit.horizon_cap, int(points), dtype=np.float32)
    survival = predict_survival(fit, split, grid)
    return np.trapz(survival, grid, axis=1).astype(np.float64)
