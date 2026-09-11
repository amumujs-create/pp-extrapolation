#!/usr/bin/env python3
"""Post-test development of causal-backtest CCMR v1.7 on luminosity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import luminosity_temperature_shift_ccmr_v161 as lum
from pp_extrapolation.causal_backtest import apply_causal_backtest_gate
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

OUT = ROOT / "results/ccmr_v17_luminosity_development"


def all_rows(frame, temperatures):
    correction, context, y, groups = [], [], [], []
    progress_values, origins, targets = [], [], []
    selected = frame[frame["celsius"].astype(float).isin(temperatures)]
    for (celsius, unit), rows in selected.groupby(["celsius", "unit"]):
        rows = rows.sort_values("hours")
        time = rows["hours"].to_numpy(dtype=float)
        health = rows["luminosity"].to_numpy(dtype=float)
        scale = float(np.median(health[:3]))
        if len(rows) < 4 or not np.isfinite(scale) or abs(scale) <= 1e-12:
            continue
        health = health / scale
        for index in range(lum.HISTORY - 1, len(rows) - 1):
            dt1 = max(float(time[index] - time[index - 1]), 1e-12)
            dt2 = max(float(time[index] - time[index - 2]), 1e-12)
            slope1 = float((health[index] - health[index - 1]) / dt1)
            slope2 = float((health[index] - health[index - 2]) / dt2)
            curvature = slope1 - slope2
            window = health[index - lum.HISTORY + 1:index + 1]
            correction.append([slope1, slope2, curvature])
            context.append([
                float(health[index]),
                slope1,
                slope2,
                curvature,
                float(window.mean()),
                float(window.std()),
                float(min((index + 1) / lum.HISTORY, 1.0)),
            ])
            y.append(float(health[index + 1]))
            groups.append(lum.unit_key(celsius, unit))
            progress_values.append(float(index / (len(rows) - 1)))
            origins.append(index)
            targets.append(index + 1)
    return {
        "correction": np.asarray(correction),
        "context": np.asarray(context),
        "y": np.asarray(y),
        "groups": np.asarray(groups),
        "progress": np.asarray(progress_values),
        "origin": np.asarray(origins),
        "target": np.asarray(targets),
    }


def score(rows, prediction, evaluate):
    y = rows["y"][evaluate]
    groups = rows["groups"][evaluate]
    anchor = rows["context"][evaluate, 0]
    pred = prediction[evaluate]
    regret = regret_summary(raw_unit_regret(y, groups, anchor, pred))
    model = regression_metrics(y, pred, groups)
    baseline = regression_metrics(y, anchor, groups)
    return {
        "model": model,
        "persistence": baseline,
        "raw_regret": {
            "mean": regret[0],
            "cvar20": regret[1],
            "maximum": regret[2],
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite v1.7 development")
    frame = next(iter(pyreadr.read_r(str(lum.DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    temperatures = sorted(frame["celsius"].astype(float).unique())
    split = {
        "train": temperatures[2:],
        "validation": temperatures[1:2],
        "test": temperatures[:1],
    }
    train = lum.make_rows(frame, split["train"], "train")
    validation = lum.make_rows(
        frame, split["validation"], "validation"
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
    output = {}
    for mode, interval in (
        ("validation", (0.40, 0.60)),
        ("test", (0.75, 1.0)),
    ):
        rows = all_rows(frame, split[mode])
        candidate, base_gate = predict_consensus_residual(
            model,
            rows["correction"],
            rows["context"],
            rows["context"][:, 0],
        )
        evaluate = (
            (rows["progress"] >= interval[0])
            & (rows["progress"] <= interval[1])
        )
        prediction, causal = apply_causal_backtest_gate(
            candidate,
            rows["context"][:, 0],
            rows["y"],
            rows["groups"],
            rows["origin"],
            rows["target"],
            evaluate,
            required_wins=5,
        )
        result = score(rows, prediction, evaluate)
        result["gate"] = {
            "base_candidate_fraction": float(np.mean(
                base_gate["active"][evaluate]
            )),
            "causal_active_fraction": float(np.mean(
                causal["active"][evaluate]
            )),
        }
        output[mode] = result
    validation_regret = output["validation"]["raw_regret"]
    validation_approved = bool(
        validation_regret["mean"] <= 0
        and validation_regret["cvar20"] <= 0.01
        and validation_regret["maximum"] <= 0.02
    )
    result_path.write_text(json.dumps({
        "status": "post-test causal-backtest development",
        "required_consecutive_shadow_wins": 5,
        "validation_approved": validation_approved,
        "splits": split,
        **output,
    }, indent=2) + "\n")
    print(json.dumps({
        "validation_approved": validation_approved,
        "validation": {
            "rmse": output["validation"]["model"]["pooled"]["rmse"],
            "baseline": output["validation"]["persistence"]["pooled"]["rmse"],
            "regret": output["validation"]["raw_regret"],
            "gate": output["validation"]["gate"],
        },
        "test": {
            "rmse": output["test"]["model"]["pooled"]["rmse"],
            "baseline": output["test"]["persistence"]["pooled"]["rmse"],
            "regret": output["test"]["raw_regret"],
            "gate": output["test"]["gate"],
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
