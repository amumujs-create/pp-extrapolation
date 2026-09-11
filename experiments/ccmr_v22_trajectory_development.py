#!/usr/bin/env python3
"""Non-holdout development of the CCMR v2.2 small-cohort route."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_trajectory_development import (
    causal_prediction,
    load_development,
    score,
    selected,
    subset,
)
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.small_cohort_route import select_ccmr_v22_route

OUT = ROOT / "results/ccmr_v22_trajectory_development"
V20 = ROOT / "results/ccmr_v20_trajectory_development/results.json"


def run_one(train, validation, test, intervals, v20):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    model = fit_causal_dynamics_bank(
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
    )
    validation_candidate, validation_evidence = (
        predict_causal_dynamics_bank(
            model,
            validation["correction"],
            validation["context"],
            validation["context"][:, 0],
        )
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    causal_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    cautious_metrics = score(
        validation, validation_gated, validation_choose
    )
    units = len(np.unique(
        validation["groups"][validation_choose]
    ))
    route = select_ccmr_v22_route(
        model, units, causal_coverage, cautious_metrics
    )
    test_candidate, test_evidence = predict_causal_dynamics_bank(
        model,
        test["correction"],
        test["context"],
        test["context"][:, 0],
    )
    test_gated, test_causal = causal_prediction(
        test_candidate, test, test_choose
    )
    if route.route in ("stable_bank", "small_crossfit_bank"):
        deployed = test_candidate
        coverage = float(np.mean(
            test_evidence["active"][test_choose]
        ))
    elif route.route == "cautious_causal":
        deployed = test_gated
        coverage = float(np.mean(
            test_causal["active"][test_choose]
        ))
    else:
        deployed = test["context"][:, 0].copy()
        coverage = 0.0
    test_metrics = score(test, deployed, test_choose)
    anchor = test["context"][test_choose, 0]
    prediction = deployed[test_choose]
    fallback = prediction == anchor
    fallback_error = (
        float(np.max(np.abs(
            prediction[fallback] - anchor[fallback]
        )))
        if np.any(fallback) else 0.0
    )
    return {
        "route": asdict(route),
        "validation_units": units,
        "validation_causal_coverage": causal_coverage,
        "model": {
            "selected_candidate": model.selected_candidate,
            "expert_weights": model.expert_weights.tolist(),
            "deployment_mass": model.deployment_mass,
            "validation_macro_improvement": (
                model.validation_macro_improvement
            ),
            "validation_active_fraction": (
                model.validation_active_fraction
            ),
            "validation_raw_regret": {
                "mean": model.validation_mean_regret,
                "cvar20": model.validation_cvar_regret,
                "maximum": model.validation_max_regret,
            },
        },
        "test": {
            "coverage": coverage,
            "fallback_error": fallback_error,
            **test_metrics,
        },
        "ablation": {
            "v20_deployed": v20["test"],
            "v22_raw_bank": {
                "coverage": float(np.mean(
                    test_evidence["active"][test_choose]
                )),
                **score(test, test_candidate, test_choose),
            },
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite v2.2 development")
    v20 = json.loads(V20.read_text())["cohorts"]
    cohorts = {}
    for name, parts in load_development().items():
        cohorts[name] = run_one(*parts, v20[name])
        print(name, flush=True)
    false_accepts = sum(
        value["route"]["approved"]
        and value["test"]["pooled_improvement"] < 0
        for value in cohorts.values()
    )
    max_regret = max(
        value["test"]["raw_regret"]["maximum"]
        for value in cohorts.values()
    )
    v20_wins = sum(
        value["test"]["model"]["pooled"]["rmse"]
        < value["ablation"]["v20_deployed"]["model"]["pooled"]["rmse"]
        for value in cohorts.values()
    )
    promoted = bool(
        false_accepts == 0
        and max_regret <= 0.02
        and v20_wins >= 1
        and all(
            value["test"]["fallback_error"] == 0
            for value in cohorts.values()
        )
    )
    payload = {
        "status": "non-holdout CCMR v2.2 development complete",
        "holdouts_loaded": False,
        "cohorts": cohorts,
        "summary": {
            "cohorts": len(cohorts),
            "false_accepts": false_accepts,
            "maximum_test_raw_regret": max_regret,
            "strict_v20_wins": v20_wins,
            "exact_fallback_all": all(
                value["test"]["fallback_error"] == 0
                for value in cohorts.values()
            ),
            "promoted": promoted,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
