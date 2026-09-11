#!/usr/bin/env python3
"""One-shot CCMR v1.9 replication on Multi-Stage battery RPT capacity."""
from __future__ import annotations

import hashlib
import json
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
from pp_extrapolation.regime_router import select_ccmr_v19_regime_route
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

DATA = ROOT / "data/external/multistage_rpt/rpt_capacity_timefixed.csv"
OUT = ROOT / "results/multistage_rpt_ccmr_v19_amended2"
HISTORY = 3
HORIZON = 1


def load_trajectories():
    frame = pd.read_csv(DATA)
    trajectories = {}
    for unit, group in frame.groupby("cell"):
        group = group.dropna(subset=["capacity_ah"]).sort_values(
            "rpt_sequence"
        ).drop_duplicates("rpt_sequence", keep="last")
        capacity = group["capacity_ah"].to_numpy(dtype=float)
        if len(capacity) < 10 or capacity[0] <= 0:
            continue
        trajectories[str(unit)] = capacity / capacity[0]
    return trajectories


def split_units(trajectories):
    units = sorted(
        trajectories,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    train_end = int(np.floor(0.60 * len(units)))
    validation_end = train_end + int(np.floor(0.20 * len(units)))
    return {
        "train": units[:train_end],
        "validation": units[train_end:validation_end],
        "test": units[validation_end:],
    }


def make_rows(trajectories, units, interval=None):
    correction, context, truth, groups = [], [], [], []
    progresses, origins, targets = [], [], []
    for unit in units:
        health = trajectories[unit]
        for index in range(HISTORY - 1, len(health) - HORIZON):
            progress = float(index / (len(health) - 1))
            if interval is not None and not interval[0] <= progress <= interval[1]:
                continue
            slope1 = float(health[index] - health[index - 1])
            slope2 = float((health[index] - health[index - 2]) / 2)
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
                1.0,
            ])
            truth.append(float(health[index + HORIZON]))
            groups.append(unit)
            progresses.append(progress)
            origins.append(index)
            targets.append(index + HORIZON)
    return {
        "correction": np.asarray(correction, dtype=float),
        "context": np.asarray(context, dtype=float),
        "y": np.asarray(truth, dtype=float),
        "groups": np.asarray(groups),
        "progress": np.asarray(progresses, dtype=float),
        "origin": np.asarray(origins, dtype=int),
        "target": np.asarray(targets, dtype=int),
    }


def select(rows, interval):
    return (
        (rows["progress"] >= interval[0])
        & (rows["progress"] <= interval[1])
    )


def score(rows, prediction, chosen):
    y = rows["y"][chosen]
    groups = rows["groups"][chosen]
    anchor = rows["context"][chosen, 0]
    prediction = prediction[chosen]
    model = regression_metrics(y, prediction, groups)
    persistence = regression_metrics(y, anchor, groups)
    model_macro = np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ])
    anchor_macro = np.mean([
        value["rmse"] for value in persistence["per_unit"].values()
    ])
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


def causal_gate(candidate, rows, chosen):
    gated, audit = apply_causal_backtest_gate(
        candidate,
        rows["context"][:, 0],
        rows["y"],
        rows["groups"],
        rows["origin"],
        rows["target"],
        chosen,
        required_wins=5,
    )
    return gated, float(np.mean(audit["active"][chosen]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite Multi-Stage one-shot")
    trajectories = load_trajectories()
    split = split_units(trajectories)
    train = make_rows(trajectories, split["train"], (0.0, 0.30))
    validation_fit = make_rows(
        trajectories, split["validation"], (0.40, 0.60)
    )
    validation = make_rows(trajectories, split["validation"])
    test = make_rows(trajectories, split["test"])
    validation_select = select(validation, (0.40, 0.60))
    test_select = select(test, (0.75, 0.90))
    admissible = bool(
        len(trajectories) >= 72
        and all(len(value) >= 10 for value in trajectories.values())
        and np.sum(validation_select) >= 30
        and np.sum(test_select) >= 20
        and np.min(test["progress"][test_select])
        > np.max(train["progress"])
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "eligible_cells": len(trajectories),
            "lengths": {
                unit: len(value) for unit, value in trajectories.items()
            },
            "validation_origins": int(np.sum(validation_select)),
            "test_origins": int(np.sum(test_select)),
        }, indent=2) + "\n")
        return

    model = fit_consensus_residual(
        train["correction"], train["context"], train["y"],
        train["groups"], train["context"][:, 0],
        validation_fit["correction"], validation_fit["context"],
        validation_fit["y"], validation_fit["groups"],
        validation_fit["context"][:, 0],
        max_ensemble_folds=20,
    )
    route = select_ccmr_v19_regime_route(
        model, len(split["validation"])
    )
    validation_candidate, validation_base = predict_consensus_residual(
        model, validation["correction"], validation["context"],
        validation["context"][:, 0],
    )
    test_candidate, test_base = predict_consensus_residual(
        model, test["correction"], test["context"],
        test["context"][:, 0],
    )
    if route.route == "stable_base":
        validation_deployed = validation_candidate
        test_deployed = test_candidate
        validation_approved = True
        validation_coverage = float(
            np.mean(validation_base["active"][validation_select])
        )
    else:
        validation_gated, validation_coverage = causal_gate(
            validation_candidate, validation, validation_select
        )
        gated_metrics = score(
            validation, validation_gated, validation_select
        )
        validation_approved = bool(
            validation_coverage >= 0.10
            and gated_metrics["pooled_improvement"] >= 0.005
            and gated_metrics["macro_improvement"] >= 0.005
            and gated_metrics["raw_regret"]["mean"] <= 0.0
            and gated_metrics["raw_regret"]["cvar20"] <= 0.01
            and gated_metrics["raw_regret"]["maximum"] <= 0.02
        )
        test_gated, _ = causal_gate(
            test_candidate, test, test_select
        )
        validation_deployed = (
            validation_gated if validation_approved
            else validation["context"][:, 0].copy()
        )
        test_deployed = (
            test_gated if validation_approved
            else test["context"][:, 0].copy()
        )

    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        candidate=test_candidate[test_select],
        routed=test_deployed[test_select],
        deployed=test_deployed[test_select],
        persistence=test["context"][test_select, 0],
        truth=test["y"][test_select],
        groups=test["groups"][test_select],
        origin=test["origin"][test_select],
        target=test["target"][test_select],
        progress=test["progress"][test_select],
        route=np.asarray([route.route]),
        validation_approved=np.asarray([validation_approved]),
    )

    validation_metrics = score(
        validation, validation_deployed, validation_select
    )
    test_metrics = score(test, test_deployed, test_select)
    anchor = test["context"][test_select, 0]
    deployed = test_deployed[test_select]
    active = deployed != anchor
    coverage = float(np.mean(active))
    fallback_error = float(np.max(np.abs(
        deployed[~active] - anchor[~active]
    ))) if np.any(~active) else 0.0
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
        "status": "Multi-Stage RPT v1.9 one-shot complete",
        "evidence_grade": "selection-conditioned independent dataset",
        "protocol": "protocols/MULTISTAGE_RPT_CCMR_V19_PROTOCOL.md",
        "source_summary_sha256": hashlib.sha256(
            DATA.read_bytes()
        ).hexdigest(),
        "eligible_cells": len(trajectories),
        "trajectory_lengths": {
            unit: len(value) for unit, value in trajectories.items()
        },
        "split": split,
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
                test["progress"][test_select]
                > np.max(train["progress"])
            )),
        },
        "route": asdict(route),
        "model": {
            key: value for key, value in asdict(model).items()
            if not isinstance(value, np.ndarray)
        },
        "validation": {
            "approved": validation_approved,
            "active_fraction": validation_coverage,
            **validation_metrics,
        },
        "test": {
            "base_active_fraction": float(
                np.mean(test_base["active"][test_select])
            ),
            "deployed_active_fraction": coverage,
            "fallback_replay_error": fallback_error,
            **test_metrics,
        },
        "confirmatory_success": success,
        "performance_success_registry_eligible": success,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "eligible_cells": len(trajectories),
        "rows": payload["rows"],
        "route": payload["route"],
        "validation": {
            "approved": validation_approved,
            "pooled_improvement": validation_metrics[
                "pooled_improvement"
            ],
            "macro_improvement": validation_metrics[
                "macro_improvement"
            ],
            "raw_regret": validation_metrics["raw_regret"],
        },
        "test": {
            "r2": test_metrics["model"]["pooled"]["r2"],
            "pooled_improvement": test_metrics["pooled_improvement"],
            "macro_improvement": test_metrics["macro_improvement"],
            "raw_regret": test_metrics["raw_regret"],
            "coverage": coverage,
        },
        "confirmatory_success": success,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
