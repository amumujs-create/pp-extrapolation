#!/usr/bin/env python3
"""Retrospective PP-X distance-indexed prior-set consensus experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from ppx_v12_continuous_portfolio_development import expert_portfolios
from pp_extrapolation import regression_metrics
from pp_extrapolation.prior_set_consensus import (
    apply_prior_set_consensus,
    crossfit_prior_set_consensus,
)
from pp_extrapolation.risk_budgeted_prior import (
    fit_support_scale,
    group_mse,
    support_distance,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results" / "ppx_v11_complete_structure_ablation"
CONTINUOUS = ROOT / "results" / "ppx_v12_continuous_portfolio_development"
OUT = ROOT / "results" / "prior_set_consensus_development"
TRUSTS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
SETTINGS = {
    "epsilon": 0.02,
    "cvar_fraction": 0.20,
    "minimum_groups_per_shell": 2,
    "disagreement_quantile": 0.90,
    "minimum_relative_gain": 0.0,
}


def positive_scale(values):
    values = np.asarray(values)
    positive = values[values > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def evaluate(name, prepared, source, predictions, continuous_predictions):
    train, validation, test, _, _ = prepared
    support = fit_support_scale(train["x"])
    validation_distance = support_distance(validation["x"], support)
    test_distance = support_distance(test["x"], support)
    scale = positive_scale(validation_distance)
    validation_distance /= scale
    test_distance /= scale

    validation_all = expert_portfolios(
        name, source["cohorts"][name], predictions, "validation"
    )
    test_all = expert_portfolios(
        name, source["cohorts"][name], predictions, "test"
    )
    validation_fallback, test_fallback = validation_all[0], test_all[0]
    validation_experts, test_experts = validation_all[1:], test_all[1:]

    validation_losses = [
        float(np.mean(group_mse(
            validation["y"], prediction, validation["groups"]
        )))
        for prediction in validation_experts
    ]
    best_index = int(np.argmin(validation_losses))
    arms = {
        "fallback": test_fallback,
        "best_single_validation": test_experts[best_index],
        "continuous_portfolio": continuous_predictions[name],
    }
    decisions = {}
    for arm, nested, edges in (
        ("global_prior_set", False, (0.0, np.inf)),
        ("free_shell_prior_set", False, None),
        ("nested_shell_prior_set", True, None),
    ):
        result = crossfit_prior_set_consensus(
            validation["y"],
            validation["groups"],
            validation_fallback,
            validation_experts,
            validation_distance,
            nested=nested,
            edges=edges,
            **SETTINGS,
        )
        prediction, active, width = apply_prior_set_consensus(
            test_experts, test_fallback, test_distance, result.policy
        )
        arms[arm] = prediction
        decisions[arm] = {
            **asdict(result),
            "oof_prediction": None,
            "test_active_fraction": float(np.mean(active)),
            "finite_width_fraction": float(np.mean(np.isfinite(width))),
        }
    return {
        "distance_scale": scale,
        "candidate_trusts": TRUSTS[1:],
        "best_single_index": best_index,
        "best_single_trust": TRUSTS[best_index + 1],
        "validation_group_mse": validation_losses,
        "decisions": decisions,
        "test": {
            arm: regression_metrics(test["y"], prediction, test["groups"])
            for arm, prediction in arms.items()
        },
    }, arms


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite prior-set experiment")
    source = json.loads((SOURCE / "results.json").read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    continuous_predictions = np.load(
        CONTINUOUS / "predictions.npz", allow_pickle=False
    )
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        result, cohort_arrays = evaluate(
            name, prepared, source, predictions, continuous_predictions
        )
        results[name] = result
        arrays.update({
            f"{name}_{arm}": value
            for arm, value in cohort_arrays.items()
        })
        print(name, json.dumps({
            arm: metrics["pooled"]["r2"]
            for arm, metrics in result["test"].items()
        }), flush=True)
    target.write_text(json.dumps({
        "status": "opened retrospective development; not confirmation",
        "model": "distance-indexed set-valued prior consensus",
        "protocol": "protocols/PRIOR_SET_CONSENSUS_DEVELOPMENT_PROTOCOL.md",
        "settings": SETTINGS,
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
