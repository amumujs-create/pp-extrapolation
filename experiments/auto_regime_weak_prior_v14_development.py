#!/usr/bin/env python3
"""Retrospective auto-regime weak-prior PP-X v1.4 evaluation."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from risk_budgeted_prior_residual_development import portfolios
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.regime_prior_bank import (
    fit_auto_regime_prior,
    predict_auto_regime_prior,
    regime_probabilities,
)
from pp_extrapolation.stability_first import regret_summary, unit_regret
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
FIXED = ROOT / "results/risk_budgeted_prior_residual_development/results.json"
STABLE = ROOT / "results/stability_first_rbpr_v13_development/results.json"
OUT = ROOT / "results/auto_regime_weak_prior_v14_development"
TRUSTS = np.array([0.02, 0.05, 0.10, 0.20, 0.40])


def occupancy(probability):
    labels = np.argmax(probability, axis=1)
    return [int(np.sum(labels == index)) for index in range(
        probability.shape[1]
    )]


def replay(name, prepared, cohort, predictions, fixed, stable, n_regimes=3):
    train, validation, test, _, _ = prepared
    val_baseline, val_priors, _ = portfolios(
        name, cohort, predictions, "validation"
    )
    test_baseline, test_priors, _ = portfolios(
        name, cohort, predictions, "test"
    )
    val_experts = np.stack([val_priors[float(trust)] for trust in TRUSTS])
    test_experts = np.stack([test_priors[float(trust)] for trust in TRUSTS])
    decision = fit_auto_regime_prior(
        train["x"], validation["x"], validation["y"], validation["groups"],
        val_baseline, val_experts, TRUSTS, n_regimes=n_regimes,
    )
    validation_prediction = predict_auto_regime_prior(
        decision, validation["x"], val_baseline, val_experts, TRUSTS
    )
    test_prediction = predict_auto_regime_prior(
        decision, test["x"], test_baseline, test_experts, TRUSTS
    )
    hard_test_prediction = predict_auto_regime_prior(
        decision, test["x"], test_baseline, test_experts, TRUSTS, hard=True
    )
    test_regret = regret_summary(unit_regret(
        test["y"], test["groups"], test_baseline, test_prediction
    ))
    hard_test_regret = regret_summary(unit_regret(
        test["y"], test["groups"], test_baseline, hard_test_prediction
    ))
    val_probability = regime_probabilities(
        decision.regime_map, validation["x"]
    )
    test_probability = regime_probabilities(decision.regime_map, test["x"])
    return {
        "decision": {
            "common_scale": decision.common_scale,
            "accepted_regimes": decision.accepted_regimes,
            "validation_mean_regret": decision.validation_mean_regret,
            "validation_cvar_regret": decision.validation_cvar_regret,
            "validation_max_regret": decision.validation_max_regret,
            "regime_decisions": [
                asdict(item) for item in decision.regime_decisions
            ],
        },
        "regime_occupancy": {
            "train_fitted_regimes": len(decision.regime_map.centroids),
            "validation_rows": occupancy(val_probability),
            "test_rows": occupancy(test_probability),
            "validation_mean_max_probability": float(np.mean(
                np.max(val_probability, axis=1)
            )),
            "test_mean_max_probability": float(np.mean(
                np.max(test_probability, axis=1)
            )),
        },
        "validation": regression_metrics(
            validation["y"], validation_prediction, validation["groups"]
        ),
        "test": regression_metrics(
            test["y"], test_prediction, test["groups"]
        ),
        "test_regret": {
            "mean": test_regret[0],
            "cvar20": test_regret[1],
            "maximum": test_regret[2],
        },
        "hard_assignment_ablation": {
            "test": regression_metrics(
                test["y"], hard_test_prediction, test["groups"]
            ),
            "test_regret": {
                "mean": hard_test_regret[0],
                "cvar20": hard_test_regret[1],
                "maximum": hard_test_regret[2],
            },
        },
        "fallback_max_abs_error": (
            float(np.max(np.abs(test_prediction - test_baseline)))
            if decision.common_scale == 0 else None
        ),
        "comparators": {
            "baseline": regression_metrics(
                test["y"], test_baseline, test["groups"]
            ),
            "fixed_rbpr": fixed["cohorts"][name]["arms"]["local_cvar"]["test"],
            "stability_first_v13": stable["cohorts"][name]["test"],
        },
    }, {
        "prediction": test_prediction,
        "hard_prediction": hard_test_prediction,
        "regime_probability": test_probability,
        "centroids": decision.regime_map.centroids,
        "center": decision.regime_map.center,
        "scale": decision.regime_map.scale,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite auto-regime replay")
    source = json.loads((SOURCE / "results.json").read_text())
    fixed = json.loads(FIXED.read_text())
    stable = json.loads(STABLE.read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        results[name], cohort_arrays = replay(
            name, prepared, source["cohorts"][name], predictions,
            fixed, stable,
        )
        sensitivity = {}
        for n_regimes in (2, 4):
            sensitivity[f"k{n_regimes}"], extra_arrays = replay(
                name, prepared, source["cohorts"][name], predictions,
                fixed, stable, n_regimes=n_regimes,
            )
            arrays.update({
                f"{name}_k{n_regimes}_{key}": value
                for key, value in extra_arrays.items()
            })
        results[name]["regime_count_sensitivity"] = sensitivity
        arrays.update({
            f"{name}_{key}": value for key, value in cohort_arrays.items()
        })
        print(name, json.dumps({
            "decision": results[name]["decision"],
            "test": results[name]["test"]["pooled"],
            "test_regret": results[name]["test_regret"],
        }), flush=True)
    target.write_text(json.dumps({
        "status": "post-test auto-regime development; not confirmation",
        "model": "Auto-Regime Weak-Prior PP-X v1.4",
        "protocol": "protocols/AUTO_REGIME_WEAK_PRIOR_V14_PROTOCOL.md",
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "artifacts.npz", **arrays)


if __name__ == "__main__":
    main()
