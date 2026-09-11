#!/usr/bin/env python3
"""One-shot CCMR v1.6 evaluation on unopened Alloy A specimens."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

DATA = ROOT / "data/alloya_external/alloya.rda"
OUT = ROOT / "results/alloya_unopened_ccmr_v16"
HISTORY = 3


def frozen_split(labels):
    ordered = sorted(labels, key=lambda label: hashlib.sha256(
        f"alloya-ccmr-v16:{label}".encode()
    ).hexdigest())
    return {
        "train": ordered[:13],
        "validation": ordered[13:17],
        "test": ordered[17:21],
    }


def make_rows(frame, labels, mode):
    bounds = {
        "train": (0.0, 0.30),
        "validation": (0.40, 0.60),
        "test": (0.75, 1.0),
    }
    lower, upper = bounds[mode]
    correction, context, y, groups, progress_values = [], [], [], [], []
    for label in labels:
        unit = frame[frame["specimen"].astype(str) == label].sort_values(
            "megacycles"
        )
        time = unit["megacycles"].to_numpy(dtype=float)
        health = unit["inches"].to_numpy(dtype=float)
        health = health / health[0]
        for index in range(HISTORY - 1, len(unit) - 1):
            progress = float(index / (len(unit) - 1))
            if not lower <= progress <= upper:
                continue
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
            y.append(float(health[index + 1]))
            groups.append(label)
            progress_values.append(progress)
    return {
        "correction": np.asarray(correction, dtype=np.float64),
        "context": np.asarray(context, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "groups": np.asarray(groups),
        "progress": np.asarray(progress_values, dtype=np.float64),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite unopened Alloy A result")
    loaded = pyreadr.read_r(str(DATA))
    frame = next(iter(loaded.values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    labels = sorted(frame["specimen"].astype(str).unique())
    split = frozen_split(labels)
    rows = {
        mode: make_rows(frame, split[mode], mode)
        for mode in ("train", "validation", "test")
    }
    admissible = bool(
        len(labels) == 21
        and all(len(rows[mode]["y"]) > 0 for mode in rows)
        and len(rows["test"]["y"]) >= 8
        and np.min(rows["test"]["progress"])
        > np.max(rows["train"]["progress"])
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "specimens": len(labels),
            "split": split,
            "rows": {mode: len(value["y"]) for mode, value in rows.items()},
        }, indent=2) + "\n")
        return

    train, validation, test = (
        rows["train"], rows["validation"], rows["test"]
    )
    model = fit_consensus_residual(
        train["correction"],
        train["context"],
        train["y"],
        train["groups"],
        train["context"][:, 0],
        validation["correction"],
        validation["context"],
        validation["y"],
        validation["groups"],
        validation["context"][:, 0],
    )
    prediction, evidence = predict_consensus_residual(
        model,
        test["correction"],
        test["context"],
        test["context"][:, 0],
    )
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        prediction=prediction,
        persistence=test["context"][:, 0],
        truth=test["y"],
        groups=test["groups"],
        progress=test["progress"],
        active=evidence["active"],
        support_rejected=evidence["support_rejected"],
        consensus_rejected=evidence["consensus_rejected"],
    )

    anchor = test["context"][:, 0]
    model_metrics = regression_metrics(test["y"], prediction, test["groups"])
    baseline_metrics = regression_metrics(test["y"], anchor, test["groups"])
    regret = regret_summary(raw_unit_regret(
        test["y"], test["groups"], anchor, prediction
    ))
    shifted_context = (
        np.max(train["context"], axis=0, keepdims=True)
        + 100.0 * model.context_scale
    )
    replay_anchor = np.asarray([0.123456789])
    replay, _ = predict_consensus_residual(
        model,
        np.zeros((1, train["correction"].shape[1])),
        shifted_context,
        replay_anchor,
    )
    fallback_error = float(np.max(np.abs(replay - replay_anchor)))
    success = bool(
        model_metrics["pooled"]["r2"] > 0
        and model_metrics["pooled"]["rmse"]
        < baseline_metrics["pooled"]["rmse"]
        and regret[0] <= 0.0
        and regret[1] <= 0.02
        and regret[2] <= 0.05
        and fallback_error == 0.0
    )
    payload = {
        "status": "one-shot unopened external evaluation complete",
        "protocol": "protocols/ALLOYA_UNOPENED_CCMR_V16_PROTOCOL.md",
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "data_bytes": DATA.stat().st_size,
        "specimens": len(labels),
        "source_rows": len(frame),
        "split": split,
        "rows": {mode: len(value["y"]) for mode, value in rows.items()},
        "ordered_extrapolation": {
            "maximum_train_progress": float(np.max(train["progress"])),
            "minimum_validation_progress": float(
                np.min(validation["progress"])
            ),
            "minimum_test_progress": float(np.min(test["progress"])),
            "test_outside_train_fraction": float(np.mean(
                test["progress"] > np.max(train["progress"])
            )),
        },
        "model": {
            key: value for key, value in asdict(model).items()
            if not isinstance(value, np.ndarray)
        },
        "gate": {
            "active_fraction": float(np.mean(evidence["active"])),
            "support_rejected_fraction": float(np.mean(
                evidence["support_rejected"]
            )),
            "consensus_rejected_fraction": float(np.mean(
                evidence["consensus_rejected"]
            )),
            "fallback_replay_error": fallback_error,
        },
        "test": {
            "ccmr": model_metrics,
            "persistence": baseline_metrics,
            "regret": {
                "mean": regret[0],
                "cvar20": regret[1],
                "maximum": regret[2],
            },
        },
        "confirmatory_success": success,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "rows": payload["rows"],
        "split": split,
        "model": payload["model"],
        "gate": payload["gate"],
        "ccmr": model_metrics["pooled"],
        "persistence": baseline_metrics["pooled"],
        "regret": payload["test"]["regret"],
        "confirmatory_success": success,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
