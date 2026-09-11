#!/usr/bin/env python3
"""Non-holdout development and ablation for CCMR v2.1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_trajectory_development import (
    load_development,
    score,
    selected,
    subset,
)
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.stochastic_dynamics import (
    fit_stochastic_dynamics,
    predict_stochastic_dynamics,
)

OUT = ROOT / "results/ccmr_v21_trajectory_development"
V20_RESULTS = ROOT / (
    "results/ccmr_v20_trajectory_development/results.json"
)


def run_one(name, train, validation, test, intervals, v20_result):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    deterministic = fit_causal_dynamics_bank(
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
    stochastic = fit_stochastic_dynamics(
        deterministic,
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
    validation_prediction, validation_evidence = (
        predict_stochastic_dynamics(
            stochastic,
            validation["correction"],
            validation["context"],
            validation["context"][:, 0],
        )
    )
    validation_metrics = score(
        validation, validation_prediction, validation_choose
    )
    validation_coverage = float(np.mean(
        validation_evidence["active"][validation_choose]
    ))
    approved = bool(
        validation_coverage >= 0.10
        and validation_metrics["pooled_improvement"] >= 0.005
        and validation_metrics["macro_improvement"] >= 0.005
        and validation_metrics["raw_regret"]["mean"] <= 0.0
        and validation_metrics["raw_regret"]["cvar20"] <= 0.01
        and validation_metrics["raw_regret"]["maximum"] <= 0.02
    )
    stochastic_test, stochastic_evidence = (
        predict_stochastic_dynamics(
            stochastic,
            test["correction"],
            test["context"],
            test["context"][:, 0],
        )
    )
    deployed = (
        stochastic_test
        if approved else test["context"][:, 0].copy()
    )
    deterministic_test, deterministic_evidence = (
        predict_causal_dynamics_bank(
            deterministic,
            test["correction"],
            test["context"],
            test["context"][:, 0],
        )
    )
    deployed_metrics = score(test, deployed, test_choose)
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
        "rows": {
            "train": len(train["y"]),
            "validation": int(np.sum(validation_choose)),
            "test": int(np.sum(test_choose)),
        },
        "approved": approved,
        "model": {
            "conformal_quantile": stochastic.conformal_quantile,
            "deployment_mass": stochastic.deployment_mass,
            "validation_macro_improvement": (
                stochastic.validation_macro_improvement
            ),
            "validation_active_fraction": (
                stochastic.validation_active_fraction
            ),
        },
        "validation": {
            "coverage": validation_coverage,
            **validation_metrics,
        },
        "test": {
            "coverage": (
                float(np.mean(
                    stochastic_evidence["active"][test_choose]
                )) if approved else 0.0
            ),
            "fallback_error": fallback_error,
            **deployed_metrics,
        },
        "ablation": {
            "v20_deployed": v20_result["test"],
            "v20_raw_bank": {
                "coverage": float(np.mean(
                    deterministic_evidence["active"][test_choose]
                )),
                **score(
                    test, deterministic_test, test_choose
                ),
            },
            "v21_stochastic_raw": {
                "coverage": float(np.mean(
                    stochastic_evidence["active"][test_choose]
                )),
                **score(test, stochastic_test, test_choose),
            },
            "v21_without_conformal": (
                "not deployed; conformal mask is integral to frozen head"
            ),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite v2.1 development")
    v20 = json.loads(V20_RESULTS.read_text())["cohorts"]
    cohorts = {}
    for name, parts in load_development().items():
        cohorts[name] = run_one(name, *parts, v20[name])
        print(name, flush=True)
    false_accepts = sum(
        value["approved"]
        and value["test"]["pooled_improvement"] < 0
        for value in cohorts.values()
    )
    maximum_regret = max(
        value["test"]["raw_regret"]["maximum"]
        for value in cohorts.values()
    )
    strict_v20_wins = sum(
        value["test"]["model"]["pooled"]["rmse"]
        < value["ablation"]["v20_deployed"]["model"]["pooled"]["rmse"]
        for value in cohorts.values()
    )
    promoted = bool(
        false_accepts == 0
        and maximum_regret <= 0.02
        and strict_v20_wins >= 1
        and all(
            value["test"]["fallback_error"] == 0
            for value in cohorts.values()
        )
    )
    payload = {
        "status": "non-holdout CCMR v2.1 development complete",
        "holdouts_loaded": False,
        "cohorts": cohorts,
        "summary": {
            "cohorts": len(cohorts),
            "approvals": sum(
                value["approved"] for value in cohorts.values()
            ),
            "false_accepts": false_accepts,
            "maximum_test_raw_regret": maximum_regret,
            "strict_v20_wins": strict_v20_wins,
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
