"""Physics-centered stochastic residual head for CCMR v2.1."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .causal_dynamics_bank import (
    CausalDynamicsBankFit,
    predict_causal_dynamics_bank,
)
from .model import equal_group_weights
from .risk_budgeted_prior import group_mse
from .stability_first import raw_unit_regret, regret_summary


@dataclass(frozen=True)
class StochasticHeadState:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class StochasticDynamicsFit:
    deterministic_model: CausalDynamicsBankFit
    feature_center: np.ndarray
    feature_scale: np.ndarray
    target_scale: float
    full_state: StochasticHeadState
    fold_states: tuple[StochasticHeadState, ...]
    conformal_quantile: float
    deployment_mass: float
    sign_threshold: float
    dispersion_threshold: float
    validation_macro_improvement: float
    validation_mean_regret: float
    validation_cvar_regret: float
    validation_max_regret: float
    validation_active_fraction: float


class _StochasticHead(nn.Module):
    def __init__(self, feature_dim, noise_dim=4, width=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim + noise_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, features, noise):
        count = noise.shape[1]
        expanded = features[:, None, :].expand(-1, count, -1)
        values = torch.cat([expanded, noise], dim=2)
        return self.layers(values).squeeze(-1)


def _state(model):
    linear = [
        layer for layer in model.layers if isinstance(layer, nn.Linear)
    ]
    return StochasticHeadState(
        tuple(layer.weight.detach().cpu().numpy().copy() for layer in linear),
        tuple(layer.bias.detach().cpu().numpy().copy() for layer in linear),
    )


def _restore(state, feature_dim):
    model = _StochasticHead(feature_dim)
    linear = [
        layer for layer in model.layers if isinstance(layer, nn.Linear)
    ]
    with torch.no_grad():
        for layer, weight, bias in zip(
            linear, state.weights, state.biases
        ):
            layer.weight.copy_(torch.tensor(weight))
            layer.bias.copy_(torch.tensor(bias))
    return model


def _equal_group_subsample(groups, maximum=5000):
    if len(groups) <= maximum:
        return np.arange(len(groups))
    labels = np.unique(groups)
    quota, remainder = divmod(maximum, len(labels))
    chosen = []
    for position, label in enumerate(labels):
        rows = np.flatnonzero(groups == label)
        count = min(len(rows), quota + int(position < remainder))
        take = np.unique(np.round(np.linspace(
            0, len(rows) - 1, count
        )).astype(int))
        chosen.extend(rows[take].tolist())
    return np.asarray(sorted(chosen[:maximum]))


def _fixed_noise(rows, samples, noise_dim, seed):
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.standard_normal((rows, samples, noise_dim)),
        dtype=torch.float32,
    )


def _fit_head(
    train_x,
    train_target,
    train_center,
    train_groups,
    validation_x,
    validation_target,
    validation_center,
    seed,
):
    torch.manual_seed(seed)
    model = _StochasticHead(train_x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.1
    )
    x = torch.tensor(train_x, dtype=torch.float32)
    target = torch.tensor(train_target, dtype=torch.float32)
    center = torch.tensor(train_center, dtype=torch.float32)
    weights = torch.tensor(
        equal_group_weights(train_groups), dtype=torch.float32
    )
    validation_features = torch.tensor(
        validation_x, dtype=torch.float32
    )
    validation_truth = torch.tensor(
        validation_target, dtype=torch.float32
    )
    validation_physics = torch.tensor(
        validation_center, dtype=torch.float32
    )
    validation_noise = _fixed_noise(
        len(validation_x), 32, 4, seed + 1000
    )
    best = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(seed)
    for epoch in range(1, 201):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), 512):
            batch = torch.tensor(order[start:start + 512])
            noise = torch.randn(len(batch), 8, 4)
            samples = center[batch, None] + model(x[batch], noise)
            distance = torch.abs(samples - target[batch, None]).mean(1)
            diversity = torch.abs(
                samples[:, :, None] - samples[:, None, :]
            ).mean((1, 2))
            mean_loss = (
                samples.mean(1) - target[batch]
            ).square()
            loss = torch.mean(
                weights[batch]
                * (distance - 0.5 * diversity + 0.1 * mean_loss)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = (
                validation_physics[:, None]
                + model(validation_features, validation_noise)
            ).mean(1)
            validation_mse = float(torch.mean(
                (prediction - validation_truth) ** 2
            ))
        if validation_mse < best - 1e-10:
            best = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(best_state)
    return _state(model)


def _sample_state(state, standardized, center, samples=64, seed=12345):
    model = _restore(state, standardized.shape[1])
    model.eval()
    features = torch.tensor(standardized, dtype=torch.float32)
    noise = _fixed_noise(len(standardized), samples, 4, seed)
    with torch.no_grad():
        values = (
            torch.tensor(center[:, None], dtype=torch.float32)
            + model(features, noise)
        ).numpy()
    return values


def fit_stochastic_dynamics(
    deterministic_model,
    train_correction,
    train_context,
    train_y,
    train_groups,
    train_anchor,
    validation_correction,
    validation_context,
    validation_y,
    validation_groups,
    validation_anchor,
    *,
    mass_grid=np.linspace(0.0, 1.0, 101),
    sign_threshold=0.80,
    dispersion_threshold=0.75,
):
    """Fit stochastic correction and validation risk/conformal certificate."""
    train_context = np.asarray(train_context, dtype=float)
    validation_context = np.asarray(validation_context, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    validation_y = np.asarray(validation_y, dtype=float)
    train_groups = np.asarray(train_groups)
    validation_groups = np.asarray(validation_groups)
    train_anchor = np.asarray(train_anchor, dtype=float)
    validation_anchor = np.asarray(validation_anchor, dtype=float)
    feature_center = np.median(train_context, axis=0)
    feature_scale = np.quantile(
        train_context, 0.75, axis=0
    ) - np.quantile(train_context, 0.25, axis=0)
    feature_scale = np.where(
        feature_scale > 1e-12,
        feature_scale,
        np.std(train_context, axis=0),
    )
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    train_z = (train_context - feature_center) / feature_scale
    validation_z = (
        validation_context - feature_center
    ) / feature_scale
    train_physics, _ = predict_causal_dynamics_bank(
        deterministic_model,
        train_correction,
        train_context,
        train_anchor,
    )
    validation_physics, validation_evidence = (
        predict_causal_dynamics_bank(
            deterministic_model,
            validation_correction,
            validation_context,
            validation_anchor,
        )
    )
    target_scale = max(
        float(np.quantile(
            np.abs(train_y - train_anchor), 0.95
        )),
        1e-8,
    )
    train_target = (train_y - train_anchor) / target_scale
    validation_target = (
        validation_y - validation_anchor
    ) / target_scale
    train_center = (
        train_physics - train_anchor
    ) / target_scale
    validation_center = (
        validation_physics - validation_anchor
    ) / target_scale
    take = _equal_group_subsample(train_groups)
    full_state = _fit_head(
        train_z[take],
        train_target[take],
        train_center[take],
        train_groups[take],
        validation_z,
        validation_target,
        validation_center,
        42,
    )
    labels = np.unique(train_groups)
    buckets = [labels[index::5] for index in range(5)]
    fold_states = []
    for fold, held_out in enumerate(buckets):
        keep = ~np.isin(train_groups, held_out)
        fold_take = _equal_group_subsample(train_groups[keep])
        indices = np.flatnonzero(keep)[fold_take]
        fold_states.append(_fit_head(
            train_z[indices],
            train_target[indices],
            train_center[indices],
            train_groups[indices],
            validation_z,
            validation_target,
            validation_center,
            43 + fold,
        ))
    samples = _sample_state(
        full_state, validation_z, validation_center
    )
    mean = samples.mean(axis=1) * target_scale
    sigma = np.maximum(
        samples.std(axis=1) * target_scale, target_scale * 1e-3
    )
    correction = mean
    standardized_error = np.abs(
        validation_y - (validation_anchor + correction)
    ) / sigma
    conformal_quantile = float(np.quantile(
        standardized_error, 0.90
    ))
    fold_mean = np.asarray([
        _sample_state(
            state,
            validation_z,
            validation_center,
            seed=20000 + index,
        ).mean(axis=1) * target_scale
        for index, state in enumerate(fold_states)
    ])
    agreement = np.maximum(
        np.mean(fold_mean > 0, axis=0),
        np.mean(fold_mean < 0, axis=0),
    )
    median = np.median(fold_mean, axis=0)
    dispersion = np.median(
        np.abs(fold_mean - median[None, :]), axis=0
    ) / (np.abs(median) + target_scale * 0.01)
    active = (
        validation_evidence["active"]
        & (agreement >= sign_threshold)
        & (dispersion <= dispersion_threshold)
        & (np.abs(correction) > conformal_quantile * sigma)
        & np.isfinite(correction)
        & np.isfinite(sigma)
    )
    baseline_macro = float(np.mean(group_mse(
        validation_y, validation_anchor, validation_groups
    )))
    feasible = []
    for mass in mass_grid:
        prediction = validation_anchor.copy()
        prediction[active] += float(mass) * correction[active]
        summary = regret_summary(raw_unit_regret(
            validation_y,
            validation_groups,
            validation_anchor,
            prediction,
        ))
        macro = float(np.mean(group_mse(
            validation_y, prediction, validation_groups
        )))
        if summary[0] <= 0 and summary[1] <= 0.01 and summary[2] <= 0.02:
            feasible.append((macro, float(mass), *summary))
    macro, mass, mean_regret, cvar_regret, max_regret = min(
        feasible, key=lambda row: (row[0], row[1])
    )
    if macro >= baseline_macro - 1e-12:
        macro = baseline_macro
        mass = 0.0
        mean_regret = cvar_regret = max_regret = 0.0
    return StochasticDynamicsFit(
        deterministic_model=deterministic_model,
        feature_center=feature_center,
        feature_scale=feature_scale,
        target_scale=target_scale,
        full_state=full_state,
        fold_states=tuple(fold_states),
        conformal_quantile=conformal_quantile,
        deployment_mass=mass,
        sign_threshold=sign_threshold,
        dispersion_threshold=dispersion_threshold,
        validation_macro_improvement=float(
            (baseline_macro - macro) / max(baseline_macro, 1e-12)
        ),
        validation_mean_regret=mean_regret,
        validation_cvar_regret=cvar_regret,
        validation_max_regret=max_regret,
        validation_active_fraction=float(np.mean(active)),
    )


def predict_stochastic_dynamics(
    model,
    correction,
    context,
    anchor,
):
    """Predict with conformal, consensus, support, and finite exact fallback."""
    correction = np.asarray(correction, dtype=float)
    context = np.asarray(context, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    valid = (
        np.isfinite(anchor)
        & np.all(np.isfinite(correction), axis=1)
        & np.all(np.isfinite(context), axis=1)
    )
    prediction = anchor.copy()
    active = np.zeros(len(anchor), dtype=bool)
    support_rejected = np.ones(len(anchor), dtype=bool)
    consensus_rejected = np.ones(len(anchor), dtype=bool)
    conformal_rejected = np.ones(len(anchor), dtype=bool)
    if not np.any(valid) or model.deployment_mass == 0:
        return prediction, {
            "active": active,
            "invalid": ~valid,
            "support_rejected": support_rejected,
            "consensus_rejected": consensus_rejected,
            "conformal_rejected": conformal_rejected,
        }
    indices = np.flatnonzero(valid)
    context_valid = context[valid]
    correction_valid = correction[valid]
    anchor_valid = anchor[valid]
    standardized = (
        context_valid - model.feature_center
    ) / model.feature_scale
    physics, deterministic_evidence = predict_causal_dynamics_bank(
        model.deterministic_model,
        correction_valid,
        context_valid,
        anchor_valid,
    )
    center = (physics - anchor_valid) / model.target_scale
    samples = _sample_state(
        model.full_state, standardized, center
    )
    mean = samples.mean(axis=1) * model.target_scale
    sigma = np.maximum(
        samples.std(axis=1) * model.target_scale,
        model.target_scale * 1e-3,
    )
    fold_mean = np.asarray([
        _sample_state(
            state, standardized, center, seed=20000 + fold
        ).mean(axis=1) * model.target_scale
        for fold, state in enumerate(model.fold_states)
    ])
    agreement = np.maximum(
        np.mean(fold_mean > 0, axis=0),
        np.mean(fold_mean < 0, axis=0),
    )
    median = np.median(fold_mean, axis=0)
    dispersion = np.median(
        np.abs(fold_mean - median[None, :]), axis=0
    ) / (np.abs(median) + model.target_scale * 0.01)
    consensus = (
        (agreement >= model.sign_threshold)
        & (dispersion <= model.dispersion_threshold)
    )
    conformal = (
        np.abs(mean) > model.conformal_quantile * sigma
    )
    local_active = (
        deterministic_evidence["active"]
        & consensus
        & conformal
        & np.isfinite(mean)
        & np.isfinite(sigma)
    )
    prediction[indices[local_active]] = (
        anchor_valid[local_active]
        + model.deployment_mass * mean[local_active]
    )
    active[indices] = local_active
    support_rejected[indices] = (
        deterministic_evidence["support_rejected"]
    )
    consensus_rejected[indices] = ~consensus
    conformal_rejected[indices] = ~conformal
    return prediction, {
        "active": active,
        "invalid": ~valid,
        "support_rejected": support_rejected,
        "consensus_rejected": consensus_rejected,
        "conformal_rejected": conformal_rejected,
    }
