#!/usr/bin/env python3
"""One-shot CCMR v1.7 evaluation on compact LG M50T summaries."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.causal_backtest import apply_causal_backtest_gate
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

DATA = ROOT / "data/external/lg_m50t_expt1_summary"
OUT = ROOT / "results/lgm50t_expt1_ccmr_v17"
AXIS = "Charge Throughput [A h]"
TARGET = "C/10 Capacity [mA h]"
HISTORY = 3


def load_trajectories():
    trajectories = {}
    for path in sorted(DATA.glob("*.csv")):
        match = re.search(r"cell ([A-Z]) ", path.name)
        if not match:
            raise ValueError(f"cannot parse cell from {path.name}")
        frame = pd.read_csv(path, index_col=0)
        axis = frame[AXIS].to_numpy(dtype=float)
        response = frame[TARGET].to_numpy(dtype=float)
        keep = np.isfinite(axis) & np.isfinite(response)
        axis, response = axis[keep], response[keep]
        ordered = np.argsort(axis, kind="stable")
        axis, response = axis[ordered], response[ordered]
        unique = np.r_[True, np.diff(axis) > 0]
        trajectories[match.group(1)] = (axis[unique], response[unique])
    return trajectories


def split_units(trajectories):
    units = sorted(
        trajectories,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    return {
        "train": units[:5],
        "validation": units[5:7],
        "test": units[7:],
    }


def make_rows(trajectories, units, interval=None):
    correction, context, truth, groups = [], [], [], []
    progresses, origins, targets = [], [], []
    for unit in units:
        axis, response = trajectories[unit]
        scale = float(response[0])
        if len(response) < 10 or not np.isfinite(scale) or scale == 0:
            continue
        health = response / scale
        for index in range(HISTORY - 1, len(health) - 1):
            progress = float(index / (len(health) - 1))
            if interval is not None and not interval[0] <= progress <= interval[1]:
                continue
            slope1 = float(
                (health[index] - health[index - 1])
                / max(axis[index] - axis[index - 1], 1e-12)
            )
            slope2 = float(
                (health[index] - health[index - 2])
                / max(axis[index] - axis[index - 2], 1e-12)
            )
            curvature = slope1 - slope2
            window = health[index - HISTORY + 1:index + 1]
            correction.append([slope1, slope2, curvature])
            context.append([
                float(health[index]),
                slope1,
                slope2,
                curvature,
                float(window.mean()),
                float(window.std()),
                float(min((index + 1) / HISTORY, 1.0)),
            ])
            truth.append(float(health[index + 1]))
            groups.append(unit)
            progresses.append(progress)
            origins.append(index)
            targets.append(index + 1)
    return {
        "correction": np.asarray(correction, dtype=float),
        "context": np.asarray(context, dtype=float),
        "y": np.asarray(truth, dtype=float),
        "groups": np.asarray(groups),
        "progress": np.asarray(progresses, dtype=float),
        "origin": np.asarray(origins, dtype=int),
        "target": np.asarray(targets, dtype=int),
    }


def metrics(rows, prediction, select):
    y = rows["y"][select]
    groups = rows["groups"][select]
    anchor = rows["context"][select, 0]
    prediction = prediction[select]
    model = regression_metrics(y, prediction, groups)
    persistence = regression_metrics(y, anchor, groups)
    model_macro = float(np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ]))
    anchor_macro = float(np.mean([
        value["rmse"] for value in persistence["per_unit"].values()
    ]))
    raw = regret_summary(raw_unit_regret(
        y, groups, anchor, prediction
    ))
    return {
        "model": model,
        "persistence": persistence,
        "pooled_improvement": float(
            (persistence["pooled"]["rmse"] - model["pooled"]["rmse"])
            / persistence["pooled"]["rmse"]
        ),
        "macro_improvement": float(
            (anchor_macro - model_macro) / anchor_macro
        ),
        "raw_regret": {
            "mean": raw[0],
            "cvar20": raw[1],
            "maximum": raw[2],
        },
    }


def gated_prediction(model, rows, interval):
    candidate, base = predict_consensus_residual(
        model,
        rows["correction"],
        rows["context"],
        rows["context"][:, 0],
    )
    select = (
        (rows["progress"] >= interval[0])
        & (rows["progress"] <= interval[1])
    )
    gated, causal = apply_causal_backtest_gate(
        candidate,
        rows["context"][:, 0],
        rows["y"],
        rows["groups"],
        rows["origin"],
        rows["target"],
        select,
        required_wins=5,
    )
    return candidate, gated, select, {
        "base_active_fraction": float(np.mean(base["active"][select])),
        "causal_active_fraction": float(np.mean(causal["active"][select])),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite LG M50T one-shot result")
    trajectories = load_trajectories()
    split = split_units(trajectories)
    train = make_rows(trajectories, split["train"], (0.0, 0.30))
    validation_fit = make_rows(
        trajectories, split["validation"], (0.40, 0.60)
    )
    validation = make_rows(trajectories, split["validation"])
    test = make_rows(trajectories, split["test"])
    initial_test_select = test["progress"] >= 0.75
    admissible = bool(
        len(trajectories) == 9
        and all(len(trajectories[unit][0]) >= 10 for unit in trajectories)
        and len(np.unique(train["groups"])) == 5
        and len(np.unique(validation["groups"])) == 2
        and len(np.unique(test["groups"])) == 2
        and np.sum(initial_test_select) >= 4
        and np.min(test["progress"][initial_test_select])
        > np.max(train["progress"])
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "split": split,
            "lengths": {
                unit: len(values[0])
                for unit, values in trajectories.items()
            },
        }, indent=2) + "\n")
        return

    model = fit_consensus_residual(
        train["correction"],
        train["context"],
        train["y"],
        train["groups"],
        train["context"][:, 0],
        validation_fit["correction"],
        validation_fit["context"],
        validation_fit["y"],
        validation_fit["groups"],
        validation_fit["context"][:, 0],
        max_ensemble_folds=20,
    )
    _, validation_gated, validation_select, validation_gate = (
        gated_prediction(model, validation, (0.40, 0.60))
    )
    validation_metrics = metrics(
        validation, validation_gated, validation_select
    )
    validation_approved = bool(
        validation_gate["causal_active_fraction"] >= 0.10
        and validation_metrics["pooled_improvement"] >= 0.005
        and validation_metrics["macro_improvement"] >= 0.005
        and validation_metrics["raw_regret"]["mean"] <= 0.0
        and validation_metrics["raw_regret"]["cvar20"] <= 0.01
        and validation_metrics["raw_regret"]["maximum"] <= 0.02
    )
    candidate, gated, test_select, test_gate = gated_prediction(
        model, test, (0.75, 1.0)
    )
    deployed = (
        gated if validation_approved else test["context"][:, 0].copy()
    )
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        candidate=candidate[test_select],
        causal_gated=gated[test_select],
        deployed=deployed[test_select],
        persistence=test["context"][test_select, 0],
        truth=test["y"][test_select],
        groups=test["groups"][test_select],
        progress=test["progress"][test_select],
        validation_approved=np.asarray([validation_approved]),
    )
    test_metrics = metrics(test, deployed, test_select)
    coverage = (
        test_gate["causal_active_fraction"] if validation_approved else 0.0
    )
    anchor = test["context"][test_select, 0]
    rejected = deployed[test_select] == anchor
    fallback_error = float(np.max(np.abs(
        deployed[test_select][rejected] - anchor[rejected]
    ))) if np.any(rejected) else 0.0
    success = bool(
        validation_approved
        and test_metrics["model"]["pooled"]["r2"] > 0
        and test_metrics["pooled_improvement"] >= 0.005
        and test_metrics["macro_improvement"] >= 0.005
        and test_metrics["raw_regret"]["mean"] <= 0.0
        and test_metrics["raw_regret"]["cvar20"] <= 0.01
        and test_metrics["raw_regret"]["maximum"] <= 0.02
        and coverage >= 0.10
        and fallback_error == 0.0
    )
    payload = {
        "status": "one-shot LG M50T Experiment 1 evaluation complete",
        "evidence_grade": "independent with pre-score eligibility amendment",
        "protocol": "protocols/LGM50T_EXPT1_UNOPENED_CCMR_V17_PROTOCOL.md",
        "resource_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(DATA.glob("*.csv"))
        },
        "split": split,
        "trajectory_lengths": {
            unit: len(values[0]) for unit, values in trajectories.items()
        },
        "rows": {
            "train": len(train["y"]),
            "validation": int(np.sum(validation_select)),
            "test": int(np.sum(test_select)),
        },
        "ordered_extrapolation": {
            "maximum_train_progress": float(np.max(train["progress"])),
            "minimum_test_progress": float(
                np.min(test["progress"][test_select])
            ),
            "test_outside_train_fraction": float(np.mean(
                test["progress"][test_select] > np.max(train["progress"])
            )),
        },
        "model": {
            key: value for key, value in asdict(model).items()
            if not isinstance(value, np.ndarray)
        },
        "validation": {
            "approved": validation_approved,
            "gate": validation_gate,
            **validation_metrics,
        },
        "test": {
            "gate": {
                **test_gate,
                "deployed_active_fraction": coverage,
                "fallback_replay_error": fallback_error,
            },
            **test_metrics,
        },
        "confirmatory_success": success,
        "performance_success_registry_eligible": success,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "split": split,
        "rows": payload["rows"],
        "validation": {
            "approved": validation_approved,
            "gate": validation_gate,
            "pooled_improvement": validation_metrics["pooled_improvement"],
            "macro_improvement": validation_metrics["macro_improvement"],
            "raw_regret": validation_metrics["raw_regret"],
        },
        "test": {
            "r2": test_metrics["model"]["pooled"]["r2"],
            "rmse": test_metrics["model"]["pooled"]["rmse"],
            "baseline_rmse": test_metrics["persistence"]["pooled"]["rmse"],
            "pooled_improvement": test_metrics["pooled_improvement"],
            "macro_improvement": test_metrics["macro_improvement"],
            "raw_regret": test_metrics["raw_regret"],
            "gate": payload["test"]["gate"],
        },
        "confirmatory_success": success,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
