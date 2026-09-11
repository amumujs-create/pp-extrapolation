#!/usr/bin/env python3
"""One-shot OAIR v1.5 on unopened NASA discharge trajectories."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.invariant_residual import (
    fit_invariant_residual,
    predict_invariant_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import regret_summary, unit_regret

DATA = (
    ROOT / "data/nasa_pcoe_new_unopened/"
    "unopened_discharge_trajectories.npz"
)
OUT = ROOT / "results/nasa_pcoe_unopened_discharge_oair_v15"
SPLIT = {
    "train": {
        "B0025", "B0026", "B0027", "B0028", "B0033",
        "B0034", "B0036", "B0038", "B0039", "B0040",
    },
    "validation": {"B0041", "B0042", "B0043", "B0044"},
    "test": {"B0049", "B0050", "B0051", "B0052"},
}
HISTORY = 10


def make_rows(data, mode):
    limits = {
        "train": (0.0, 0.30),
        "validation": (0.40, 0.60),
        "test": (0.75, 0.85),
    }
    lower, upper = limits[mode]
    x, y, groups, episodes, progress_values = [], [], [], [], []
    keys = sorted({
        (str(cell), int(episode))
        for cell, episode in zip(data["cell"], data["episode"])
        if str(cell) in SPLIT[mode]
    })
    for cell, episode in keys:
        take = (data["cell"] == cell) & (data["episode"] == episode)
        time = data["time"][take]
        voltage = data["voltage"][take]
        current = data["current"][take]
        temperature = data["temperature"][take]
        scale = float(np.median(voltage[:5]))
        health = voltage / scale
        horizon = max(5, int(np.floor(0.10 * len(time))))
        for index in range(HISTORY - 1, len(time) - horizon):
            progress = float(index / (len(time) - 1))
            if not lower <= progress <= upper:
                continue
            window = health[index - HISTORY + 1:index + 1]
            slopes = []
            for lag in (1, 5, 9):
                start = index - lag
                dt = max(float(time[index] - time[start]), 1e-9)
                slopes.append(float((health[index] - health[start]) / dt))
            x.append([
                float(health[index]),
                *slopes,
                float(window.mean()),
                float(window.std()),
                float(current[index]),
                float(temperature[index]),
                float(min((index + 1) / HISTORY, 1.0)),
            ])
            y.append(float(health[index + horizon]))
            groups.append(cell)
            episodes.append(f"{cell}_D{episode}")
            progress_values.append(progress)
    return {
        "x": np.asarray(x, np.float32),
        "y": np.asarray(y, np.float32),
        "groups": np.asarray(groups),
        "episodes": np.asarray(episodes),
        "progress": np.asarray(progress_values),
        "episode_count": len(keys),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite unopened NASA test")
    data = np.load(DATA, allow_pickle=False)
    train = make_rows(data, "train")
    validation = make_rows(data, "validation")
    test = make_rows(data, "test")
    admissible = bool(
        set(np.unique(train["groups"])) == SPLIT["train"]
        and set(np.unique(validation["groups"])) == SPLIT["validation"]
        and set(np.unique(test["groups"])) == SPLIT["test"]
        and test["episode_count"] >= 30
        and len(test["y"]) >= 1000
    )
    if not admissible:
        result_path.write_text(json.dumps({
            "status": "inconclusive before model fitting",
            "rows": {
                "train": len(train["y"]),
                "validation": len(validation["y"]),
                "test": len(test["y"]),
            },
            "episodes": {
                "train": train["episode_count"],
                "validation": validation["episode_count"],
                "test": test["episode_count"],
            },
        }, indent=2) + "\n")
        return

    regime_columns = [1, 2, 3]
    model = fit_invariant_residual(
        train["x"][:, regime_columns],
        train["y"],
        train["groups"],
        train["x"][:, 0],
        validation["x"][:, regime_columns],
        validation["y"],
        validation["groups"],
        validation["x"][:, 0],
    )
    prediction, out_of_support = predict_invariant_residual(
        model, test["x"][:, regime_columns], test["x"][:, 0]
    )

    # Materialize predictions before any test score.
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        prediction=prediction,
        persistence=test["x"][:, 0],
        groups=test["groups"],
        episodes=test["episodes"],
        progress=test["progress"],
        out_of_support=out_of_support,
    )
    model_metrics = regression_metrics(
        test["y"], prediction, test["groups"]
    )
    persistence_metrics = regression_metrics(
        test["y"], test["x"][:, 0], test["groups"]
    )
    regret = regret_summary(unit_regret(
        test["y"], test["groups"], test["x"][:, 0], prediction
    ))
    outside_fraction = float(np.mean(
        test["progress"] > np.max(train["progress"])
    ))
    success = bool(
        outside_fraction == 1.0
        and model_metrics["pooled"]["r2"] > 0
        and model_metrics["pooled"]["rmse"]
        < persistence_metrics["pooled"]["rmse"]
        and regret[0] <= 0.02
        and regret[1] <= 0.05
        and regret[2] <= 0.10
    )
    payload = {
        "status": "one-shot unopened external evaluation complete",
        "protocol": (
            "protocols/"
            "NASA_PCOE_UNOPENED_DISCHARGE_OAIR_V15_PROTOCOL.md"
        ),
        "compact_data_bytes": DATA.stat().st_size,
        "rows": {
            "train": len(train["y"]),
            "validation": len(validation["y"]),
            "test": len(test["y"]),
        },
        "episodes": {
            "train": train["episode_count"],
            "validation": validation["episode_count"],
            "test": test["episode_count"],
        },
        "cells": {key: sorted(value) for key, value in SPLIT.items()},
        "ordered_extrapolation": {
            "maximum_train_progress": float(np.max(train["progress"])),
            "minimum_validation_progress": float(
                np.min(validation["progress"])
            ),
            "minimum_test_progress": float(np.min(test["progress"])),
            "test_outside_train_fraction": outside_fraction,
        },
        "model": {
            key: value for key, value in asdict(model).items()
            if key not in {"center", "scale", "coefficient", "support"}
        },
        "ood": {
            "rows": int(np.sum(out_of_support)),
            "fraction": float(np.mean(out_of_support)),
        },
        "test": {
            "oair": model_metrics,
            "persistence": persistence_metrics,
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
        "episodes": payload["episodes"],
        "ordered_extrapolation": payload["ordered_extrapolation"],
        "model": payload["model"],
        "ood": payload["ood"],
        "test": {
            key: value["pooled"]
            for key, value in (
                ("oair", model_metrics),
                ("persistence", persistence_metrics),
            )
        },
        "regret": payload["test"]["regret"],
        "confirmatory_success": success,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
