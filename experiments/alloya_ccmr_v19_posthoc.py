#!/usr/bin/env python3
"""Retrospective CCMR v1.9 replay on the already-opened Alloy A cohort."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from alloya_unopened_ccmr_v16 import DATA, HISTORY, frozen_split
from multistage_rpt_ccmr_v19 import causal_gate, score, select
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.regime_router import select_ccmr_v19_regime_route

OUT = ROOT / "results/alloya_ccmr_v19_posthoc"


def make_all_rows(frame, labels):
    correction, context, truth, groups = [], [], [], []
    progresses, origins, targets = [], [], []
    for label in labels:
        unit = frame[
            frame["specimen"].astype(str) == label
        ].sort_values("megacycles")
        time = unit["megacycles"].to_numpy(dtype=float)
        health = unit["inches"].to_numpy(dtype=float)
        health = health / health[0]
        for index in range(HISTORY - 1, len(unit) - 1):
            slope1 = float(
                (health[index] - health[index - 1])
                / (time[index] - time[index - 1])
            )
            slope2 = float(
                (health[index] - health[index - 2])
                / (time[index] - time[index - 2])
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
            groups.append(label)
            progresses.append(float(index / (len(unit) - 1)))
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


def subset(rows, chosen):
    return {
        key: value[chosen]
        for key, value in rows.items()
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite Alloy A v1.9 replay")
    frame = next(iter(pyreadr.read_r(str(DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    split = frozen_split(
        sorted(frame["specimen"].astype(str).unique())
    )
    full = {
        mode: make_all_rows(frame, split[mode])
        for mode in ("train", "validation", "test")
    }
    train_select = select(full["train"], (0.0, 0.30))
    validation_select = select(full["validation"], (0.40, 0.60))
    test_select = select(full["test"], (0.75, 1.0))
    train = subset(full["train"], train_select)
    validation_fit = subset(
        full["validation"], validation_select
    )
    validation = full["validation"]
    test = full["test"]

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
    route = select_ccmr_v19_regime_route(
        model, len(split["validation"])
    )
    validation_candidate, validation_base = (
        predict_consensus_residual(
            model,
            validation["correction"],
            validation["context"],
            validation["context"][:, 0],
        )
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
        validation_coverage = float(np.mean(
            validation_base["active"][validation_select]
        ))
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
            validation_gated
            if validation_approved
            else validation["context"][:, 0].copy()
        )
        test_deployed = (
            test_gated
            if validation_approved
            else test["context"][:, 0].copy()
        )

    validation_metrics = score(
        validation, validation_deployed, validation_select
    )
    test_metrics = score(test, test_deployed, test_select)
    test_anchor = test["context"][test_select, 0]
    test_prediction = test_deployed[test_select]
    active = test_prediction != test_anchor
    payload = {
        "status": "retrospective Alloy A CCMR v1.9 replay complete",
        "confirmatory": False,
        "reason": "Alloy A test was already opened under v1.6",
        "split": split,
        "rows": {
            "train": int(np.sum(train_select)),
            "validation": int(np.sum(validation_select)),
            "test": int(np.sum(test_select)),
        },
        "route": asdict(route),
        "validation": {
            "approved": validation_approved,
            "active_fraction": validation_coverage,
            **validation_metrics,
        },
        "test": {
            "base_active_fraction": float(np.mean(
                test_base["active"][test_select]
            )),
            "deployed_active_fraction": float(np.mean(active)),
            **test_metrics,
        },
    }
    np.savez_compressed(
        OUT / "predictions.npz",
        candidate=test_candidate[test_select],
        deployed=test_prediction,
        persistence=test_anchor,
        truth=test["y"][test_select],
        groups=test["groups"][test_select],
        active=active,
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "route": payload["route"],
        "validation": {
            "approved": validation_approved,
            "coverage": validation_coverage,
            "pooled_improvement": validation_metrics[
                "pooled_improvement"
            ],
        },
        "test": {
            "r2": test_metrics["model"]["pooled"]["r2"],
            "pooled_improvement": test_metrics["pooled_improvement"],
            "macro_improvement": test_metrics["macro_improvement"],
            "raw_regret": test_metrics["raw_regret"],
            "coverage": float(np.mean(active)),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
