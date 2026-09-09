"""Causal temporal encoder with a PP affine path and bounded residual."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from .model import equal_group_weights


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(F.pad(value, (2 * self.dilation, 0)))
        return value + self.dropout(F.silu(self.norm(convolved)))


class TemporalResidualPP(nn.Module):
    def __init__(self, input_dim: int, width: int, *, mode: str,
                 residual_bound: float, support_decay: float, dropout: float):
        super().__init__()
        if mode not in ("pp", "direct"):
            raise ValueError("mode must be pp or direct")
        self.mode = mode
        self.residual_bound = float(residual_bound)
        self.support_decay = float(support_decay)
        self.projection = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.ModuleList(CausalBlock(width, d, dropout) for d in (1, 2, 4, 8))
        self.residual = nn.Linear(width, 1)
        self.affine = nn.Linear(input_dim, 1)
        self.direct = nn.Linear(width, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.register_buffer("support_min", torch.zeros(input_dim))
        self.register_buffer("support_max", torch.zeros(input_dim))

    def encode(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = self.projection(sequence.transpose(1, 2))
        for block in self.blocks:
            value = block(value)
        weight = mask[:, None, :].to(value.dtype)
        return (value * weight).sum(2) / weight.sum(2).clamp_min(1.0)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(sequence, mask)
        if self.mode == "direct":
            raw = self.direct(encoded).squeeze(1)
        else:
            current = sequence[:, -1]
            below = torch.relu(self.support_min - current)
            above = torch.relu(current - self.support_max)
            distance = torch.linalg.vector_norm(below + above, dim=1)
            gate = torch.exp(-self.support_decay * distance)
            correction = self.residual_bound * torch.tanh(self.residual(encoded).squeeze(1))
            raw = self.affine(current).squeeze(1) + gate * correction
        # Smooth nonnegative output without a train-target upper bound.
        return F.softplus(raw / 0.05) * 0.05


@dataclass
class TemporalResidualFit:
    model: TemporalResidualPP
    center: np.ndarray
    scale: np.ndarray
    target_scale: float
    selection: dict


def fit_temporal_residual_pp(train: dict, validation: dict, *, seed: int,
                             mode: str = "pp", width: int = 16,
                             learning_rate: float = 5e-4,
                             weight_decay: float = 0.05,
                             residual_bound: float = 0.5,
                             support_decay: float = 0.0,
                             dropout: float = 0.0,
                             max_epochs: int = 350, patience: int = 60) -> TemporalResidualFit:
    tx = np.asarray(train["x"], np.float32)
    vx = np.asarray(validation["x"], np.float32)
    tm = np.asarray(train["mask"], bool)
    vm = np.asarray(validation["mask"], bool)
    ty = np.asarray(train["y"], np.float32)
    vy = np.asarray(validation["y"], np.float32)
    groups = np.asarray(train["groups"])
    if tx.ndim != 3 or vx.ndim != 3 or tm.shape != tx.shape[:2] or vm.shape != vx.shape[:2]:
        raise ValueError("x must be (n,time,feature) with an aligned mask")
    observed = tx[tm]
    center = observed.mean(0)
    scale = observed.std(0)
    scale[scale < 1e-6] = 1.0
    tx = ((tx - center) / scale).astype(np.float32)
    vx = ((vx - center) / scale).astype(np.float32)
    tx[~tm] = 0.0
    vx[~vm] = 0.0
    target_scale = max(float(np.quantile(ty, 0.95)), 1.0)

    torch.manual_seed(seed)
    model = TemporalResidualPP(tx.shape[2], width, mode=mode,
                               residual_bound=residual_bound,
                               support_decay=support_decay, dropout=dropout)
    current = tx[:, -1]
    with torch.no_grad():
        model.support_min.copy_(torch.tensor(current.min(0)))
        model.support_max.copy_(torch.tensor(current.max(0)))
    affine_alpha = None
    if mode == "pp":
        candidates = []
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            ridge = Ridge(alpha=alpha).fit(current, ty / target_scale)
            raw = ((vx[:, -1] @ ridge.coef_) + ridge.intercept_)
            pred = np.logaddexp(0, raw / 0.05) * 0.05 * target_scale
            candidates.append((float(np.mean((pred - vy) ** 2)), alpha, ridge))
        _, affine_alpha, ridge = min(candidates, key=lambda row: row[0])
        with torch.no_grad():
            model.affine.weight.copy_(torch.tensor(ridge.coef_[None], dtype=torch.float32))
            model.affine.bias.copy_(torch.tensor([ridge.intercept_], dtype=torch.float32))
        model.affine.requires_grad_(False)

    xt = torch.tensor(tx)
    xv = torch.tensor(vx)
    mt = torch.tensor(tm)
    mv = torch.tensor(vm)
    yt = torch.tensor(ty / target_scale)
    weights = torch.tensor(equal_group_weights(groups), dtype=torch.float32)
    parameters = list(model.projection.parameters()) + list(model.blocks.parameters())
    parameters += list(model.residual.parameters() if mode == "pp" else model.direct.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    def validation_mse() -> float:
        model.eval()
        with torch.no_grad():
            prediction = model(xv, mv).numpy() * target_scale
        return float(np.mean((prediction - vy) ** 2))

    best = validation_mse()
    best_epoch = 0
    state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "validation_rmse": best ** 0.5}]
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tx))
        for start in range(0, len(order), 256):
            index = torch.tensor(order[start:start + 256])
            prediction = model(xt[index], mt[index])
            loss = torch.mean(weights[index] * (prediction - yt[index]).square())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()
        score = validation_mse()
        history.append({"epoch": epoch, "validation_rmse": score ** 0.5})
        if score < best - 1e-8:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break
    model.load_state_dict(state)
    return TemporalResidualFit(model, center, scale, target_scale, {
        "mode": mode, "seed": seed, "width": width, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "residual_bound": residual_bound,
        "support_decay": support_decay, "dropout": dropout,
        "affine_alpha": affine_alpha, "selected_epoch": best_epoch,
        "validation_rmse": best ** 0.5, "history": history,
    })


def predict_temporal_residual_pp(fit: TemporalResidualFit, data: dict) -> np.ndarray:
    x = ((np.asarray(data["x"], np.float32) - fit.center) / fit.scale).astype(np.float32)
    mask = np.asarray(data["mask"], bool)
    x[~mask] = 0.0
    fit.model.eval()
    with torch.no_grad():
        return (fit.model(torch.tensor(x), torch.tensor(mask)).numpy() * fit.target_scale).astype(float)
