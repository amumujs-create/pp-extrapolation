#!/usr/bin/env python3
"""One-shot CCMR v2.0 on Stage-2 TP_z remainder (same-cell as Stage-1)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_trajectory_development import causal_prediction, score
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)

DATA = ROOT / "data/external/multistage_stage2_tpz/rpt_capacity.csv"
OUT = ROOT / "results/multistage_stage2_tpz_ccmr_v20"
MANIFEST = ROOT / "protocols/CCMR_V20_FROZEN_PREDICTOR_MANIFEST.json"
PROTOCOL = ROOT / "protocols/MULTISTAGE_STAGE2_TPZ_CCMR_V20_PROTOCOL.md"
HISTORY = 3
HORIZON = 1
TRAIN_INTERVAL = (0.0, 0.30)
VAL_INTERVAL = (0.40, 0.60)
TEST_INTERVAL = (0.75, 0.90)
SPLIT_SALT = "multistage-stage2-tpz-ccmr-v20:"


def load_trajectories():
    if not DATA.exists():
        raise RuntimeError(
            "missing Stage-2 TP_z extraction; run extract_multistage_stage2_tpz.py"
        )
    frame = pd.read_csv(DATA)
    trajectories = {}
    for unit, group in frame.groupby("cell"):
        if not str(unit).startswith("TP_z"):
            raise RuntimeError(f"non-TP_z cell leaked: {unit}")
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
        key=lambda value: hashlib.sha256(
            f"{SPLIT_SALT}{value}".encode()
        ).hexdigest(),
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
            if interval is not None and not (
                interval[0] <= progress <= interval[1]
            ):
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite Stage-2 TP_z one-shot")
    trajectories = load_trajectories()
    split = split_units(trajectories)
    train = make_rows(trajectories, split["train"], TRAIN_INTERVAL)
    validation = make_rows(trajectories, split["validation"])
    test = make_rows(trajectories, split["test"])
    validation_choose = select(validation, VAL_INTERVAL)
    test_choose = select(test, TEST_INTERVAL)
    validation_fit = {
        key: value[validation_choose]
        if isinstance(value, np.ndarray) else value
        for key, value in validation.items()
    }
    admissible = bool(
        len(trajectories) >= 50
        and np.sum(validation_choose) >= 20
        and np.sum(test_choose) >= 15
        and len(train["y"]) > 0
        and np.min(test["progress"][test_choose]) > np.max(train["progress"])
    )
    payload = {
        "status": "inconclusive before model fitting",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "predictor_manifest": str(MANIFEST.relative_to(ROOT)),
        "predictor_manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "eligible_cells": len(trajectories),
        "split": {key: list(value) for key, value in split.items()},
        "validation_origins": int(np.sum(validation_choose)),
        "test_origins": int(np.sum(test_choose)),
        "confirmatory": False,
        "selection_conditioned": True,
        "same_physical_units_as_stage1": True,
        "independent_cohort": False,
        "counted_sealed_auto_promote": False,
    }
    if not admissible:
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        return

    model = fit_causal_dynamics_bank(
        train["correction"], train["context"], train["y"],
        train["groups"], train["context"][:, 0],
        validation_fit["correction"], validation_fit["context"],
        validation_fit["y"], validation_fit["groups"],
        validation_fit["context"][:, 0],
    )
    validation_candidate, validation_base = predict_causal_dynamics_bank(
        model, validation["correction"], validation["context"],
        validation["context"][:, 0],
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    validation_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    stable = bool(
        len(np.unique(validation["groups"][validation_choose])) >= 10
        and model.validation_active_fraction >= 0.50
        and model.validation_macro_improvement >= 0.05
        and model.validation_mean_regret <= 0.0
        and model.validation_cvar_regret <= 0.01
        and model.validation_max_regret <= 0.02
        and validation_coverage >= 0.10
    )
    cautious_metrics = score(
        validation, validation_gated, validation_choose
    )
    cautious = bool(
        validation_coverage >= 0.10
        and cautious_metrics["pooled_improvement"] >= 0.005
        and cautious_metrics["macro_improvement"] >= 0.005
        and cautious_metrics["raw_regret"]["mean"] <= 0.0
        and cautious_metrics["raw_regret"]["cvar20"] <= 0.01
        and cautious_metrics["raw_regret"]["maximum"] <= 0.02
    )
    route = (
        "stable_bank" if stable
        else "cautious_causal" if cautious
        else "exact_fallback"
    )
    test_candidate, test_base = predict_causal_dynamics_bank(
        model, test["correction"], test["context"], test["context"][:, 0],
    )
    test_gated, test_causal = causal_prediction(
        test_candidate, test, test_choose
    )
    deployed = (
        test_candidate if stable
        else test_gated if cautious
        else test["context"][:, 0].copy()
    )
    selected_anchor = test["context"][test_choose, 0]
    selected_prediction = deployed[test_choose]
    inactive = selected_prediction == selected_anchor
    fallback_error = (
        float(np.max(np.abs(
            selected_prediction[inactive] - selected_anchor[inactive]
        )))
        if np.any(inactive) else 0.0
    )
    test_metrics = score(test, deployed, test_choose)
    coverage = (
        float(np.mean(test_base["active"][test_choose])) if stable
        else float(np.mean(test_causal["active"][test_choose])) if cautious
        else 0.0
    )
    success = bool(
        float(np.min(test["progress"][test_choose]))
        > float(np.max(train["progress"]))
        and test_metrics["model"]["pooled"]["r2"] > 0.0
        and test_metrics["pooled_improvement"] >= 0.005
        and test_metrics["macro_improvement"] >= 0.005
        and test_metrics["raw_regret"]["mean"] <= 0.0
        and test_metrics["raw_regret"]["cvar20"] <= 0.01
        and test_metrics["raw_regret"]["maximum"] <= 0.02
        and coverage >= 0.10
        and fallback_error == 0.0
        and route != "exact_fallback"
    )
    payload.update({
        "status": (
            "performance success (same-cell continued; not independent)"
            if success else "safe fallback or failure"
        ),
        "performance_success": success,
        "route": route,
        "model": {
            "selected_candidate": model.selected_candidate,
            "expert_names": [expert.name for expert in model.experts],
            "expert_weights": model.expert_weights.tolist(),
            "ridge_alpha": model.ridge_alpha,
            "deployment_mass": model.deployment_mass,
            "validation_macro_improvement": model.validation_macro_improvement,
            "validation_active_fraction": model.validation_active_fraction,
        },
        "validation": {
            "causal_shadow_coverage": validation_coverage,
            "metrics": (
                score(validation, validation_candidate, validation_choose)
                if stable else cautious_metrics
            ),
        },
        "test": {
            "deployed_coverage": coverage,
            "fallback_error": fallback_error,
            **test_metrics,
        },
    })
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        candidate=test_candidate[test_choose],
        deployed=selected_prediction,
        persistence=selected_anchor,
        truth=test["y"][test_choose],
        groups=test["groups"][test_choose].astype(str),
        origin=test["origin"][test_choose],
        target=test["target"][test_choose],
        progress=test["progress"][test_choose],
        route=np.asarray([route]),
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
