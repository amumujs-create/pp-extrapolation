#!/usr/bin/env python3
"""One-shot CCMR v1.7 on unopened HPC-to-UHPC material shift."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.causal_backtest import apply_causal_backtest_gate
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

DATA = ROOT / "data/concrete_external"
OUT = ROOT / "results/concrete_material_shift_ccmr_v17"
HISTORY = 10
SPLIT_FILES = {
    "train": ("HPC_Lab_1.zip", "HPC_Lab_2.zip", "HPC_Lab_3.zip"),
    "validation": ("HPC_Lab_4.zip",),
    "test": ("UHPC_Lab_5.zip", "UHPC_Lab_6.zip", "UHPC_Lab_7.zip"),
}


def load_trajectories(names):
    trajectories = {}
    for name in names:
        path = DATA / name
        with ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".txt"):
                    continue
                cycle, stiffness = [], []
                for line in archive.read(member).decode(
                    errors="replace"
                ).splitlines():
                    parts = line.split()
                    if len(parts) < 13:
                        continue
                    try:
                        cycle_value = float(parts[0])
                        stiffness_value = float(parts[9])
                    except ValueError:
                        continue
                    if np.isfinite(cycle_value) and np.isfinite(stiffness_value):
                        cycle.append(cycle_value)
                        stiffness.append(stiffness_value)
                cycle = np.asarray(cycle, dtype=np.float64)
                stiffness = np.asarray(stiffness, dtype=np.float64)
                if len(cycle):
                    keep = np.r_[True, np.diff(cycle) > 0]
                    cycle, stiffness = cycle[keep], stiffness[keep]
                key = f"{Path(name).stem}:{Path(member).stem}"
                trajectories[key] = (cycle, stiffness)
    return trajectories


def make_rows(trajectories, interval=None):
    correction, context, y, groups = [], [], [], []
    progress_values, origins, targets = [], [], []
    for name, (cycle, stiffness) in trajectories.items():
        if len(cycle) < 100:
            continue
        scale = float(np.median(stiffness[:10]))
        if not np.isfinite(scale) or abs(scale) <= 1e-12:
            continue
        health = stiffness / scale
        horizon = max(5, int(np.floor(0.05 * len(health))))
        for index in range(HISTORY - 1, len(health) - horizon):
            progress = float(index / (len(health) - 1))
            if interval is not None and not interval[0] <= progress <= interval[1]:
                continue
            slopes = []
            for lag in (1, 3, 9):
                dt = max(float(cycle[index] - cycle[index - lag]), 1e-12)
                slopes.append(float(
                    (health[index] - health[index - lag]) / dt
                ))
            curvature = slopes[0] - slopes[2]
            window = health[index - HISTORY + 1:index + 1]
            correction.append([*slopes, curvature])
            context.append([
                float(health[index]),
                *slopes,
                curvature,
                float(window.mean()),
                float(window.std()),
                float(min((index + 1) / HISTORY, 1.0)),
            ])
            y.append(float(health[index + horizon]))
            groups.append(name)
            progress_values.append(progress)
            origins.append(index)
            targets.append(index + horizon)
    return {
        "correction": np.asarray(correction, dtype=np.float64),
        "context": np.asarray(context, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "groups": np.asarray(groups),
        "progress": np.asarray(progress_values, dtype=np.float64),
        "origin": np.asarray(origins, dtype=np.int64),
        "target": np.asarray(targets, dtype=np.int64),
    }


def evaluate_metrics(rows, prediction, select):
    truth = rows["y"][select]
    groups = rows["groups"][select]
    anchor = rows["context"][select, 0]
    prediction = prediction[select]
    model = regression_metrics(truth, prediction, groups)
    baseline = regression_metrics(truth, anchor, groups)
    raw = regret_summary(raw_unit_regret(
        truth, groups, anchor, prediction
    ))
    model_macro = float(np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ]))
    baseline_macro = float(np.mean([
        value["rmse"] for value in baseline["per_unit"].values()
    ]))
    return {
        "model": model,
        "persistence": baseline,
        "pooled_improvement": float(
            (baseline["pooled"]["rmse"] - model["pooled"]["rmse"])
            / baseline["pooled"]["rmse"]
        ),
        "macro_improvement": float(
            (baseline_macro - model_macro) / baseline_macro
        ),
        "raw_regret": {
            "mean": raw[0],
            "cvar20": raw[1],
            "maximum": raw[2],
        },
    }


def causal_prediction(model, rows, interval):
    candidate, base_evidence = predict_consensus_residual(
        model,
        rows["correction"],
        rows["context"],
        rows["context"][:, 0],
    )
    select = (
        (rows["progress"] >= interval[0])
        & (rows["progress"] <= interval[1])
    )
    gated, causal_evidence = apply_causal_backtest_gate(
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
        "base_active_fraction": float(np.mean(
            base_evidence["active"][select]
        )),
        "causal_active_fraction": float(np.mean(
            causal_evidence["active"][select]
        )),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite concrete one-shot result")
    trajectories = {
        mode: load_trajectories(names)
        for mode, names in SPLIT_FILES.items()
    }
    unit_counts = {
        mode: len(values) for mode, values in trajectories.items()
    }
    train = make_rows(trajectories["train"], (0.0, 0.30))
    validation_fit = make_rows(
        trajectories["validation"], (0.40, 0.60)
    )
    validation = make_rows(trajectories["validation"])
    test = make_rows(trajectories["test"])
    test_select = (
        (test["progress"] >= 0.75) & (test["progress"] <= 0.90)
    )
    admissible = bool(
        unit_counts["train"] >= 20
        and unit_counts["validation"] >= 5
        and unit_counts["test"] >= 20
        and np.sum(test_select) >= 500
        and np.min(test["progress"][test_select])
        > np.max(train["progress"])
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "unit_counts": unit_counts,
            "rows": {
                "train": len(train["y"]),
                "validation": len(validation_fit["y"]),
                "test": int(np.sum(test_select)),
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
        causal_prediction(model, validation, (0.40, 0.60))
    )
    validation_metrics = evaluate_metrics(
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

    test_candidate, test_gated, test_select, test_gate = causal_prediction(
        model, test, (0.75, 0.90)
    )
    deployed = (
        test_gated if validation_approved
        else test["context"][:, 0].copy()
    )
    # Seal every test prediction before computing test metrics.
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        candidate=test_candidate[test_select],
        causal_gated=test_gated[test_select],
        deployed=deployed[test_select],
        persistence=test["context"][test_select, 0],
        truth=test["y"][test_select],
        groups=test["groups"][test_select],
        progress=test["progress"][test_select],
        validation_approved=np.asarray([validation_approved]),
    )

    test_metrics = evaluate_metrics(test, deployed, test_select)
    deployed_coverage = (
        test_gate["causal_active_fraction"] if validation_approved else 0.0
    )
    replay_anchor = test["context"][test_select, 0]
    rejected = deployed[test_select] == replay_anchor
    fallback_error = float(np.max(np.abs(
        deployed[test_select][rejected] - replay_anchor[rejected]
    ))) if np.any(rejected) else 0.0
    success = bool(
        validation_approved
        and test_metrics["model"]["pooled"]["r2"] > 0
        and test_metrics["pooled_improvement"] >= 0.005
        and test_metrics["macro_improvement"] >= 0.005
        and test_metrics["raw_regret"]["mean"] <= 0.0
        and test_metrics["raw_regret"]["cvar20"] <= 0.01
        and test_metrics["raw_regret"]["maximum"] <= 0.02
        and deployed_coverage >= 0.10
        and fallback_error == 0.0
    )
    payload = {
        "status": "one-shot concrete material-shift evaluation complete",
        "protocol": "protocols/CONCRETE_MATERIAL_SHIFT_CCMR_V17_PROTOCOL.md",
        "resource_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(DATA.glob("*.zip"))
        },
        "unit_counts": unit_counts,
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
                "deployed_active_fraction": deployed_coverage,
                "fallback_replay_error": fallback_error,
            },
            **test_metrics,
        },
        "confirmatory_success": success,
        "second_sealed_success_eligible": success,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "unit_counts": unit_counts,
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
