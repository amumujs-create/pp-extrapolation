#!/usr/bin/env python3
"""Post-test LED development of consensus-certified minimax residual v1.6."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import led_strong_extrapolation_v14 as led
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

OUT = ROOT / "results/ccmr_v16_led_development"
CORRECTION_COLUMNS = [1, 2, 3]
CONTEXT_COLUMNS = [0, 1, 2, 3, 4, 5, 7]
VARIANTS = {
    "full_ccmr": {},
    "scalable_group20": {"max_ensemble_folds": 20},
    "no_consensus": {
        "sign_threshold": 0.5,
        "dispersion_threshold": 1e12,
    },
    "minimax_only": {
        "sign_threshold": 0.5,
        "dispersion_threshold": 1e12,
        "constant_context": True,
    },
}


def metrics(split, prediction):
    anchor = split["x"][:, 0]
    regret = regret_summary(raw_unit_regret(
        split["y"], split["groups"], anchor, prediction
    ))
    return {
        "model": regression_metrics(split["y"], prediction, split["groups"]),
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
    results = {}
    for name, settings in VARIANTS.items():
        settings = dict(settings)
        constant_context = settings.pop("constant_context", False)
        if constant_context:
            train_context = np.zeros((len(train["y"]), 1))
            validation_context = np.zeros((len(validation["y"]), 1))
            test_context = np.zeros((len(test["y"]), 1))
        else:
            train_context = train["x"][:, CONTEXT_COLUMNS]
            validation_context = validation["x"][:, CONTEXT_COLUMNS]
            test_context = test["x"][:, CONTEXT_COLUMNS]
        model = fit_consensus_residual(
            train["x"][:, CORRECTION_COLUMNS],
            train_context,
            train["y"],
            train["groups"],
            train["x"][:, 0],
            validation["x"][:, CORRECTION_COLUMNS],
            validation_context,
            validation["y"],
            validation["groups"],
            validation["x"][:, 0],
            **settings,
        )
        prediction, evidence = predict_consensus_residual(
            model,
            test["x"][:, CORRECTION_COLUMNS],
            test_context,
            test["x"][:, 0],
        )
        score = metrics(test, prediction)
        score["fit"] = {
            key: value for key, value in asdict(model).items()
            if not isinstance(value, np.ndarray)
        }
        score["gate"] = {
            "active_fraction": float(np.mean(evidence["active"])),
            "consensus_rejected_fraction": float(np.mean(
                evidence["consensus_rejected"]
            )),
            "support_rejected_fraction": float(np.mean(
                evidence["support_rejected"]
            )),
        }
        results[name] = score
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite v1.6 development")
    cohorts = {
        f"Data_ID_{data_id}": evaluate(data_id)
        for data_id in led.DATA_IDS
    }
    target.write_text(json.dumps({
        "status": "post-test LED development and ablation",
        "cohorts": cohorts,
    }, indent=2) + "\n")
    for cohort, variants in cohorts.items():
        print(cohort, json.dumps({
            name: {
                "rmse": value["model"]["pooled"]["rmse"],
                "baseline": value["persistence"]["pooled"]["rmse"],
                "max_regret": value["regret"]["maximum"],
                "active": value["gate"]["active_fraction"],
                "mass": value["fit"]["deployment_mass"],
            }
            for name, value in variants.items()
        }), flush=True)


if __name__ == "__main__":
    main()
