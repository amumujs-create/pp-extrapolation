#!/usr/bin/env python3
"""Final sequential CCMR v1.8 replication on RADAR NMC cyclic cells."""
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
from pp_extrapolation.regime_router import select_ccmr_regime_route
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

DATA = ROOT / "data/external/radar_nmc_result_v2/extracted"
OUT = ROOT / "results/radar_nmc_cyclic_ccmr_v18"
HISTORY = 20
HORIZON = 10


def false_like(values):
    return values.astype(str).str.strip().str.lower().isin(
        {"0", "false", "f"}
    )


def load_trajectories():
    trajectories = {}
    paths = sorted(DATA.rglob("cell_eocv2_*.csv"))
    for path in paths:
        frame = pd.read_csv(path, sep=";")
        required = {
            "age_type", "cyc_condition", "cyc_charged",
            "num_cycles_op", "timestamp_s", "cap_aged_est_Ah",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"schema mismatch in {path.name}")
        frame = frame[
            (pd.to_numeric(frame["age_type"], errors="coerce") == 2)
            & (pd.to_numeric(
                frame["cyc_condition"], errors="coerce"
            ) == 1)
            & false_like(frame["cyc_charged"])
        ].copy()
        frame["cycle"] = pd.to_numeric(
            frame["num_cycles_op"], errors="coerce"
        )
        frame["timestamp"] = pd.to_numeric(
            frame["timestamp_s"], errors="coerce"
        )
        frame["capacity"] = pd.to_numeric(
            frame["cap_aged_est_Ah"], errors="coerce"
        )
        frame = frame.dropna(
            subset=["cycle", "timestamp", "capacity"]
        ).sort_values(["cycle", "timestamp"])
        frame = frame.drop_duplicates("cycle", keep="last")
        if len(frame) < 500:
            continue
        cycle = frame["cycle"].to_numpy(dtype=float)
        capacity = frame["capacity"].to_numpy(dtype=float)
        if np.any(np.diff(cycle) <= 0):
            raise ValueError(f"non-increasing cycle in {path.name}")
        initial = float(np.max(capacity[:5]))
        health = pd.Series(capacity / initial).rolling(
            5, min_periods=1
        ).median().to_numpy()
        unit = path.stem.removeprefix("cell_eocv2_")
        trajectories[unit] = (cycle, health)
    return trajectories, paths


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
        cycle, health = trajectories[unit]
        for index in range(HISTORY - 1, len(health) - HORIZON):
            progress = float(index / (len(health) - 1))
            if interval is not None and not interval[0] <= progress <= interval[1]:
                continue
            slopes = [
                float(
                    (health[index] - health[index - lag])
                    / max(cycle[index] - cycle[index - lag], 1e-12)
                )
                for lag in (1, 5, 20)
            ]
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


def interval_select(rows, interval):
    return (
        (rows["progress"] >= interval[0])
        & (rows["progress"] <= interval[1])
    )


def score(rows, prediction, select):
    y = rows["y"][select]
    groups = rows["groups"][select]
    anchor = rows["context"][select, 0]
    prediction = prediction[select]
    model_metrics = regression_metrics(y, prediction, groups)
    persistence = regression_metrics(y, anchor, groups)
    model_macro = np.mean([
        value["rmse"] for value in model_metrics["per_unit"].values()
    ])
    anchor_macro = np.mean([
        value["rmse"] for value in persistence["per_unit"].values()
    ])
    raw = regret_summary(raw_unit_regret(
        y, groups, anchor, prediction
    ))
    return {
        "model": model_metrics,
        "persistence": persistence,
        "pooled_improvement": float(
            (persistence["pooled"]["rmse"]
             - model_metrics["pooled"]["rmse"])
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


def causal_route_prediction(candidate, rows, select):
    gated, gate = apply_causal_backtest_gate(
        candidate,
        rows["context"][:, 0],
        rows["y"],
        rows["groups"],
        rows["origin"],
        rows["target"],
        select,
        required_wins=5,
    )
    return gated, {
        "causal_active_fraction": float(np.mean(gate["active"][select]))
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite RADAR NMC one-shot result")
    trajectories, source_paths = load_trajectories()
    split = split_units(trajectories)
    train = make_rows(trajectories, split["train"], (0.0, 0.30))
    validation_fit = make_rows(
        trajectories, split["validation"], (0.40, 0.60)
    )
    validation = make_rows(trajectories, split["validation"])
    test = make_rows(trajectories, split["test"])
    validation_select = interval_select(validation, (0.40, 0.60))
    test_select = interval_select(test, (0.75, 0.90))
    admissible = bool(
        len(trajectories) >= 100
        and len(split["train"]) >= 60
        and len(split["validation"]) >= 20
        and len(split["test"]) >= 20
        and np.sum(test_select) >= 5000
        and np.min(test["progress"][test_select])
        > np.max(train["progress"])
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "source_files": len(source_paths),
            "eligible_cells": len(trajectories),
            "split_sizes": {
                key: len(value) for key, value in split.items()
            },
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
    route = select_ccmr_regime_route(model)
    validation_candidate, validation_base = predict_consensus_residual(
        model,
        validation["correction"],
        validation["context"],
        validation["context"][:, 0],
    )
    test_candidate, test_base = predict_consensus_residual(
        model,
        test["correction"],
        test["context"],
        test["context"][:, 0],
    )
    if route.route == "stable_base":
        validation_deployed = validation_candidate
        test_deployed = test_candidate
        validation_approved = True
        validation_gate_coverage = float(
            np.mean(validation_base["active"][validation_select])
        )
    else:
        validation_gated, validation_gate = causal_route_prediction(
            validation_candidate, validation, validation_select
        )
        validation_metrics_for_gate = score(
            validation, validation_gated, validation_select
        )
        validation_gate_coverage = validation_gate[
            "causal_active_fraction"
        ]
        validation_approved = bool(
            validation_gate_coverage >= 0.10
            and validation_metrics_for_gate["pooled_improvement"] >= 0.005
            and validation_metrics_for_gate["macro_improvement"] >= 0.005
            and validation_metrics_for_gate["raw_regret"]["mean"] <= 0.0
            and validation_metrics_for_gate["raw_regret"]["cvar20"] <= 0.01
            and validation_metrics_for_gate["raw_regret"]["maximum"] <= 0.02
        )
        test_gated, _ = causal_route_prediction(
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
    deployed_selected = test_deployed[test_select]
    active = deployed_selected != anchor
    coverage = float(np.mean(active))
    rejected = ~active
    fallback_error = float(np.max(np.abs(
        deployed_selected[rejected] - anchor[rejected]
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
        "status": "final RADAR NMC sequential replication complete",
        "evidence_grade": "selection-conditioned independent dataset",
        "protocol": "protocols/RADAR_NMC_CYCLIC_CCMR_V18_PROTOCOL.md",
        "source_archive_sha256": hashlib.sha256(
            (DATA.parent / "10.35097-1969.tar").read_bytes()
        ).hexdigest(),
        "source_files": len(source_paths),
        "eligible_cells": len(trajectories),
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
                test["progress"][test_select] > np.max(train["progress"])
            )),
        },
        "route": asdict(route),
        "model": {
            key: value for key, value in asdict(model).items()
            if not isinstance(value, np.ndarray)
        },
        "validation": {
            "approved": validation_approved,
            "active_fraction": validation_gate_coverage,
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
        "split_sizes": {
            key: len(value) for key, value in split.items()
        },
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
