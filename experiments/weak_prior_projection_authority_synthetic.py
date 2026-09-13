#!/usr/bin/env python3
"""Synthetic mechanism test for learned weak-prior projection authority."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.weak_prior_projection import (  # noqa: E402
    apply_weak_prior_projection,
    fit_authority_head,
    predict_authority,
)

OUT = ROOT / "results" / "weak_prior_projection_authority_synthetic_v1"
PROTOCOL = (
    "protocols/WEAK_PRIOR_PROJECTION_AUTHORITY_SYNTHETIC_PROTOCOL.md"
)
DEVELOPMENT_TASKS = 180
CONFIRMATION_TASKS = 60
SOURCE_X = np.linspace(0.0, 0.60, 64)
TAIL_X = np.linspace(0.60, 1.0, 49)[1:]
GRID = np.linspace(0.0, 1.0, 401)
AUTHORITY_GRID = np.linspace(0.0, 1.0, 11)
FEATURE_NAMES = (
    "direct_pseudo_rmse",
    "hard_pseudo_rmse",
    "relative_projection_gain",
    "minimum_slope_violation_rate",
    "negative_prediction_rate",
    "source_curvature",
    "source_linear_residual_scale",
    "pseudo_tail_distance",
    "projection_correction_magnitude",
    "source_tail_slope",
)
ABLATION_FEATURES = (0, 3, 4, 5, 6, 7, 9)


class BaseMLP(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def function_parameters(seed):
    rng = np.random.default_rng(seed)
    valid = bool(seed % 2 == 0)
    return {
        "intercept": rng.uniform(0.5, 1.5),
        "slope": rng.uniform(0.8, 2.5),
        "curve": rng.uniform(0.0, 1.2),
        "smooth": rng.uniform(0.0, 0.20),
        "reversal": 0.0 if valid else rng.uniform(5.0, 14.0),
        "valid_generator": valid,
        "noise": rng.uniform(0.01, 0.06),
    }


def truth(x, parameters):
    x = np.asarray(x)
    value = (
        parameters["intercept"]
        + parameters["slope"] * x
        + parameters["curve"] * x**2
        + parameters["smooth"] * (1.0 - np.cos(np.pi * x))
        - parameters["reversal"] * np.maximum(x - 0.60, 0.0) ** 2
    )
    return np.maximum(value, 0.0)


def fit_base(x, y, seed):
    center = float(np.mean(x))
    scale = max(float(np.std(x)), 1e-8)
    target_center = float(np.mean(y))
    target_scale = max(float(np.std(y)), 1e-8)
    tx = torch.as_tensor(
        ((x - center) / scale)[:, None], dtype=torch.float32
    )
    ty = torch.as_tensor(
        (y - target_center) / target_scale, dtype=torch.float32
    )
    torch.manual_seed(seed)
    model = BaseMLP()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.02
    )
    best = float("inf")
    state = copy.deepcopy(model.state_dict())
    for _ in range(500):
        estimate = model(tx)
        loss = torch.mean((estimate - ty) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "center": center,
        "scale": scale,
        "target_center": target_center,
        "target_scale": target_scale,
    }


def predict_base(fit, x):
    values = torch.as_tensor(
        ((np.asarray(x) - fit["center"]) / fit["scale"])[:, None],
        dtype=torch.float32,
    )
    with torch.no_grad():
        prediction = fit["model"](values).numpy()
    return prediction * fit["target_scale"] + fit["target_center"]


def source_slope(x, y):
    count = max(8, int(np.ceil(0.30 * len(x))))
    coefficient = np.polyfit(x[-count:], y[-count:], 1)[0]
    return max(float(coefficient), 0.0)


def project_ray(raw_boundary, raw_tail, derivative, spacing):
    ray = np.concatenate([[raw_boundary], raw_tail])[None, :]
    increment = np.asarray([0.50 * derivative * spacing])
    _, feasible = apply_weak_prior_projection(
        ray, np.ones(1), increment
    )
    return feasible[0, 1:], increment[0]


def best_authority(raw, hard, y):
    losses = np.asarray([
        np.sqrt(np.mean((raw + alpha * (hard - raw) - y) ** 2))
        for alpha in AUTHORITY_GRID
    ])
    return float(AUTHORITY_GRID[int(np.argmin(losses))])


def task(seed):
    parameters = function_parameters(seed)
    rng = np.random.default_rng(seed + 100_000)
    source_y = truth(SOURCE_X, parameters) + rng.normal(
        0.0, parameters["noise"], len(SOURCE_X)
    )
    inner = SOURCE_X <= 0.45
    pseudo = ~inner
    inner_fit = fit_base(SOURCE_X[inner], source_y[inner], seed)
    inner_boundary = float(predict_base(inner_fit, [SOURCE_X[inner][-1]])[0])
    pseudo_raw = predict_base(inner_fit, SOURCE_X[pseudo])
    inner_slope = source_slope(SOURCE_X[inner], source_y[inner])
    pseudo_hard, pseudo_increment = project_ray(
        inner_boundary,
        pseudo_raw,
        inner_slope,
        float(np.mean(np.diff(SOURCE_X[pseudo]))),
    )
    pseudo_scale = max(float(np.std(source_y[pseudo])), 1e-8)
    direct_rmse = float(np.sqrt(np.mean(
        (pseudo_raw - source_y[pseudo]) ** 2
    )))
    hard_rmse = float(np.sqrt(np.mean(
        (pseudo_hard - source_y[pseudo]) ** 2
    )))
    ray = np.concatenate([[inner_boundary], pseudo_raw])
    differences = np.diff(ray)
    quadratic = np.polyfit(SOURCE_X[inner], source_y[inner], 2)
    linear_prediction = np.polyval(
        np.polyfit(SOURCE_X[inner], source_y[inner], 1), SOURCE_X[inner]
    )
    features = np.asarray([
        direct_rmse / pseudo_scale,
        hard_rmse / pseudo_scale,
        (direct_rmse - hard_rmse) / max(direct_rmse, 1e-8),
        float(np.mean(differences < pseudo_increment - 1e-8)),
        float(np.mean(ray < 0)),
        float(abs(quadratic[0]) / max(np.std(source_y[inner]), 1e-8)),
        float(np.std(source_y[inner] - linear_prediction) /
              max(np.std(source_y[inner]), 1e-8)),
        float((SOURCE_X[pseudo][0] - SOURCE_X[inner][-1]) /
              max(np.ptp(SOURCE_X[inner]), 1e-8)),
        float(np.mean(np.abs(pseudo_hard - pseudo_raw)) / pseudo_scale),
        float(inner_slope / max(np.std(source_y[inner]), 1e-8)),
    ])
    pseudo_authority = best_authority(
        pseudo_raw, pseudo_hard, source_y[pseudo]
    )

    final_fit = fit_base(SOURCE_X, source_y, seed + 1)
    boundary_raw = float(predict_base(final_fit, [SOURCE_X[-1]])[0])
    tail_raw = predict_base(final_fit, TAIL_X)
    final_slope = source_slope(SOURCE_X, source_y)
    tail_hard, minimum_increment = project_ray(
        boundary_raw, tail_raw, final_slope, float(np.mean(np.diff(TAIL_X)))
    )
    tail_y = truth(TAIL_X, parameters)
    oracle_authority = best_authority(tail_raw, tail_hard, tail_y)
    true_grid = truth(GRID, parameters)
    weak_prior_valid = bool(np.all(
        np.diff(true_grid) >= -1e-10
    ))
    return {
        "seed": seed,
        "generator_valid": parameters["valid_generator"],
        "weak_prior_valid": weak_prior_valid,
        "features": features,
        "pseudo_authority": pseudo_authority,
        "oracle_authority": oracle_authority,
        "tail_raw": tail_raw,
        "tail_hard": tail_hard,
        "tail_y": tail_y,
        "minimum_increment": minimum_increment,
    }


def evaluate_task(row, full_authority, ablated_authority):
    raw = row["tail_raw"]
    hard = row["tail_hard"]
    y = row["tail_y"]
    authorities = {
        "direct_mlp": 0.0,
        "hard_projection": 1.0,
        "pseudo_tail_authority": row["pseudo_authority"],
        "ablated_authority_head": float(ablated_authority),
        "full_authority_head": float(full_authority),
        "tail_oracle_non_deployable": row["oracle_authority"],
    }
    arms = {}
    direct_rmse = float(np.sqrt(np.mean((raw - y) ** 2)))
    for name, authority in authorities.items():
        prediction = raw + authority * (hard - raw)
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        arms[name] = {
            "authority": authority,
            "rmse": rmse,
            "relative_rmse_reduction_vs_direct": (
                direct_rmse - rmse
            ) / max(direct_rmse, 1e-12),
        }
    return arms


def summarize(rows, arm, valid=None):
    selected = [
        row for row in rows
        if valid is None or row["weak_prior_valid"] == valid
    ]
    rmse = np.asarray([row["arms"][arm]["rmse"] for row in selected])
    direct = np.asarray([
        row["arms"]["direct_mlp"]["rmse"] for row in selected
    ])
    reduction = (direct - rmse) / np.maximum(direct, 1e-12)
    harm = -reduction
    count = max(1, int(np.ceil(0.20 * len(harm))))
    authority = np.asarray([
        row["arms"][arm]["authority"] for row in selected
    ])
    return {
        "tasks": len(selected),
        "mean_rmse": float(np.mean(rmse)),
        "mean_relative_rmse_reduction": float(np.mean(reduction)),
        "task_win_fraction": float(np.mean(reduction > 1e-12)),
        "worst20_mean_relative_harm": float(
            np.mean(np.sort(harm)[-count:])
        ),
        "mean_authority": float(np.mean(authority)),
        "fraction_partial_authority": float(np.mean(
            (authority > 1e-6) & (authority < 1 - 1e-6)
        )),
    }


def main():
    torch.set_num_threads(2)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite weak-prior synthetic run")
    print("generating development tasks", flush=True)
    development = [
        task(10_000 + index) for index in range(DEVELOPMENT_TASKS)
    ]
    features = np.asarray([row["features"] for row in development])
    oracle = np.asarray([row["oracle_authority"] for row in development])
    full_head = fit_authority_head(features, oracle, seed=20260912)
    ablated_head = fit_authority_head(
        features[:, ABLATION_FEATURES], oracle, seed=20260912
    )

    # Heads are frozen before confirmation task tails are scored.
    print("generating held-out confirmation tasks", flush=True)
    confirmation = [
        task(20_000 + index) for index in range(CONFIRMATION_TASKS)
    ]
    confirmation_features = np.asarray([
        row["features"] for row in confirmation
    ])
    full_authority = predict_authority(full_head, confirmation_features)
    ablated_authority = predict_authority(
        ablated_head, confirmation_features[:, ABLATION_FEATURES]
    )
    rows = []
    for row, full, ablated in zip(
        confirmation, full_authority, ablated_authority
    ):
        rows.append({
            "seed": row["seed"],
            "generator_valid": row["generator_valid"],
            "weak_prior_valid": row["weak_prior_valid"],
            "features": row["features"].tolist(),
            "pseudo_authority": row["pseudo_authority"],
            "oracle_authority": row["oracle_authority"],
            "minimum_increment": row["minimum_increment"],
            "arms": evaluate_task(row, full, ablated),
        })
    arm_names = tuple(rows[0]["arms"])
    payload = {
        "status": "held-out synthetic mechanism confirmation",
        "owner": "박진서",
        "protocol": PROTOCOL,
        "development_tasks": DEVELOPMENT_TASKS,
        "confirmation_tasks": CONFIRMATION_TASKS,
        "feature_names": FEATURE_NAMES,
        "ablated_feature_indices": ABLATION_FEATURES,
        "full_head": {
            "best_epoch": full_head.best_epoch,
            "validation_loss": full_head.validation_loss,
        },
        "ablated_head": {
            "best_epoch": ablated_head.best_epoch,
            "validation_loss": ablated_head.validation_loss,
        },
        "oracle_authority_correlation": float(np.corrcoef(
            full_authority,
            np.asarray([row["oracle_authority"] for row in confirmation]),
        )[0, 1]),
        "summaries": {
            "all": {
                arm: summarize(rows, arm) for arm in arm_names
            },
            "valid_weak_prior": {
                arm: summarize(rows, arm, True) for arm in arm_names
            },
            "violated_weak_prior": {
                arm: summarize(rows, arm, False) for arm in arm_names
            },
        },
        "tasks": rows,
        "guardrails": [
            "Confirmation-tail outcomes never enter authority features.",
            "Authority heads are fit before confirmation tasks are scored.",
            "The same base MLP predictions are used in every arm.",
            "The tail oracle is nondeployable.",
            "Synthetic confirmation does not replace real-data validation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "oracle_authority_correlation": payload[
            "oracle_authority_correlation"
        ],
        "all": payload["summaries"]["all"],
    }, indent=2))


if __name__ == "__main__":
    main()
