#!/usr/bin/env python3
"""One-shot CCMR-L v1.6.1 temperature-shift luminosity evaluation."""
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

DATA = ROOT / "data/luminosity_external/luminosity.rda"
OUT = ROOT / "results/luminosity_temperature_shift_ccmr_v161"
HISTORY = 3


def unit_key(celsius, unit):
    return f"{float(celsius):.12g}:{str(unit)}"


def make_rows(frame, temperatures, mode):
    bounds = {
        "train": (0.0, 0.30),
        "validation": (0.40, 0.60),
        "test": (0.75, 1.0),
    }
    lower, upper = bounds[mode]
    correction, context, y, groups, progress_values = [], [], [], [], []
    take = frame["celsius"].astype(float).isin(temperatures)
    selected = frame[take]
    for (celsius, unit), rows in selected.groupby(["celsius", "unit"]):
        rows = rows.sort_values("hours")
        time = rows["hours"].to_numpy(dtype=float)
        health = rows["luminosity"].to_numpy(dtype=float)
        scale = float(np.median(health[:3]))
        if len(rows) < 4 or not np.isfinite(scale) or abs(scale) <= 1e-12:
            continue
        health = health / scale
        for index in range(HISTORY - 1, len(rows) - 1):
            progress = float(index / (len(rows) - 1))
            if not lower <= progress <= upper:
                continue
            dt1 = max(float(time[index] - time[index - 1]), 1e-12)
            dt2 = max(float(time[index] - time[index - 2]), 1e-12)
            slope1 = float((health[index] - health[index - 1]) / dt1)
            slope2 = float((health[index] - health[index - 2]) / dt2)
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
            groups.append(unit_key(celsius, unit))
            progress_values.append(progress)
    return {
        "correction": np.asarray(correction, dtype=np.float64),
        "context": np.asarray(context, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "groups": np.asarray(groups),
        "progress": np.asarray(progress_values, dtype=np.float64),
    }


def macro_rmse(metrics):
    return float(np.mean([
        row["rmse"] for row in metrics["per_unit"].values()
    ]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite luminosity one-shot result")
    frame = next(iter(pyreadr.read_r(str(DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    temperatures = sorted(frame["celsius"].astype(float).unique())
    split = {
        "train": temperatures[2:],
        "validation": temperatures[1:2],
        "test": temperatures[:1],
    }
    data = {
        mode: make_rows(frame, split[mode], mode)
        for mode in ("train", "validation", "test")
    }
    unit_counts = {
        mode: len(np.unique(rows["groups"]))
        for mode, rows in data.items()
    }
    admissible = bool(
        len(temperatures) >= 3
        and unit_counts["train"] >= 20
        and unit_counts["validation"] >= 5
        and unit_counts["test"] >= 5
        and len(data["test"]["y"]) >= 20
        and np.min(data["test"]["progress"])
        > np.max(data["train"]["progress"])
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "temperatures": temperatures,
            "unit_counts": unit_counts,
            "rows": {
                mode: len(rows["y"]) for mode, rows in data.items()
            },
        }, indent=2) + "\n")
        return

    train, validation, test = (
        data["train"], data["validation"], data["test"]
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
        max_ensemble_folds=20,
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
    model_macro = macro_rmse(model_metrics)
    baseline_macro = macro_rmse(baseline_metrics)
    pooled_improvement = (
        baseline_metrics["pooled"]["rmse"] - model_metrics["pooled"]["rmse"]
    ) / baseline_metrics["pooled"]["rmse"]
    macro_improvement = (baseline_macro - model_macro) / baseline_macro
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
    active_fraction = float(np.mean(evidence["active"]))
    success = bool(
        model_metrics["pooled"]["r2"] > 0
        and pooled_improvement >= 0.005
        and macro_improvement >= 0.005
        and regret[0] <= 0.0
        and regret[1] <= 0.02
        and regret[2] <= 0.05
        and active_fraction >= 0.10
        and fallback_error == 0.0
    )
    safe_fallback = bool(
        not success
        and regret[0] <= 0.0
        and regret[1] <= 0.02
        and regret[2] <= 0.05
        and fallback_error == 0.0
    )
    payload = {
        "status": "one-shot temperature-shift evaluation complete",
        "protocol": (
            "protocols/"
            "LUMINOSITY_TEMPERATURE_SHIFT_CCMR_V161_PROTOCOL.md"
        ),
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "data_bytes": DATA.stat().st_size,
        "source_rows": len(frame),
        "temperatures_celsius": temperatures,
        "split": split,
        "unit_counts": unit_counts,
        "rows": {mode: len(rows["y"]) for mode, rows in data.items()},
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
            "active_fraction": active_fraction,
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
            "ccmr_unit_macro_rmse": model_macro,
            "persistence_unit_macro_rmse": baseline_macro,
            "pooled_rmse_improvement": pooled_improvement,
            "unit_macro_rmse_improvement": macro_improvement,
            "raw_regret": {
                "mean": regret[0],
                "cvar20": regret[1],
                "maximum": regret[2],
            },
        },
        "confirmatory_success": success,
        "safe_fallback": safe_fallback,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "temperatures": temperatures,
        "unit_counts": unit_counts,
        "rows": payload["rows"],
        "model": payload["model"],
        "gate": payload["gate"],
        "ccmr": model_metrics["pooled"],
        "persistence": baseline_metrics["pooled"],
        "pooled_improvement": pooled_improvement,
        "macro_improvement": macro_improvement,
        "raw_regret": payload["test"]["raw_regret"],
        "confirmatory_success": success,
        "safe_fallback": safe_fallback,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
