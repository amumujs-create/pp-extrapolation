#!/usr/bin/env python3
"""Post-test OAIR v1.5 replay on frozen strong LED extrapolation splits."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import led_strong_extrapolation_v14 as led
from pp_extrapolation.invariant_residual import (
    fit_invariant_residual,
    predict_invariant_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import regret_summary, unit_regret

OUT = ROOT / "results/oair_v15_led_strong_development"
V14 = ROOT / "results/led_strong_extrapolation_v14/results.json"
REGIME_COLUMNS = [1, 2, 3]


def score(split, prediction, anchor):
    regret = regret_summary(unit_regret(
        split["y"], split["groups"], anchor, prediction
    ))
    return {
        "oair": regression_metrics(
            split["y"], prediction, split["groups"]
        ),
        "persistence": regression_metrics(
            split["y"], anchor, split["groups"]
        ),
        "regret": {
            "mean": regret[0],
            "cvar20": regret[1],
            "maximum": regret[2],
        },
    }


def evaluate(data_id):
    cells = led.load_cells(data_id)
    ids = led.split_ids(cells, data_id)
    time_scale = led.train_time_scale(cells, ids["train"])
    train = led.make_rows(cells, ids["train"], time_scale, "train")
    validation = led.make_rows(
        cells, ids["validation"], time_scale, "validation"
    )
    test = led.make_rows(cells, ids["test"], time_scale, "test")
    model = fit_invariant_residual(
        train["x"][:, REGIME_COLUMNS],
        train["y"],
        train["groups"],
        train["x"][:, 0],
        validation["x"][:, REGIME_COLUMNS],
        validation["y"],
        validation["groups"],
        validation["x"][:, 0],
    )
    prediction, out_of_support = predict_invariant_residual(
        model, test["x"][:, REGIME_COLUMNS], test["x"][:, 0]
    )
    metrics = score(test, prediction, test["x"][:, 0])
    pooled = metrics["oair"]["pooled"]
    baseline = metrics["persistence"]["pooled"]
    regret = metrics["regret"]
    success = bool(
        pooled["r2"] > 0
        and pooled["rmse"] < baseline["rmse"]
        and regret["mean"] <= 0.02
        and regret["cvar20"] <= 0.05
        and regret["maximum"] <= 0.10
    )
    return {
        "rows": {
            "train": len(train["y"]),
            "validation": len(validation["y"]),
            "test": len(test["y"]),
        },
        "test_outside_train_progress_fraction": float(np.mean(
            test["progress"] > np.max(train["progress"])
        )),
        "model": {
            key: value for key, value in asdict(model).items()
            if key not in {"center", "scale", "coefficient", "support"}
        },
        "ood_rows": int(np.sum(out_of_support)),
        "test": metrics,
        "success": success,
    }, {
        "prediction": prediction,
        "persistence": test["x"][:, 0],
        "truth": test["y"],
        "groups": test["groups"],
        "ood": out_of_support,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite OAIR v1.5 replay")
    v14 = json.loads(V14.read_text())
    cohorts, arrays = {}, {}
    for data_id in led.DATA_IDS:
        name = f"Data_ID_{data_id}"
        result, cohort_arrays = evaluate(data_id)
        result["absolute_output_v14"] = v14["cohorts"][name]["test"]["model"]
        cohorts[name] = result
        arrays.update({
            f"D{data_id}_{key}": value
            for key, value in cohort_arrays.items()
        })
        print(name, json.dumps({
            "oair": result["test"]["oair"]["pooled"],
            "persistence": result["test"]["persistence"]["pooled"],
            "regret": result["test"]["regret"],
            "success": result["success"],
        }), flush=True)
    overall = all(result["success"] for result in cohorts.values())
    target.write_text(json.dumps({
        "status": "post-test architecture repair development complete",
        "protocol": "protocols/OAIR_V15_POSTTEST_DEVELOPMENT_PROTOCOL.md",
        "overall_success": overall,
        "cohorts": cohorts,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    print("overall_success", overall, flush=True)


if __name__ == "__main__":
    main()
