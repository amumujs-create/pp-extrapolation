#!/usr/bin/env python3
"""Retrospective heterogeneous structural prior-set experiment."""
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
from pp_extrapolation.risk_budgeted_prior import (
    fit_support_scale,
    group_mse,
    support_distance,
)
from pp_extrapolation.structural_prior_set import (
    apply_structural_prior_set,
    crossfit_structural_prior_set,
    fit_affine_prior,
    fit_history_rate_prior,
    fit_monotone_health_prior,
    health_distance,
    predict_affine_prior,
    predict_history_rate_prior,
    predict_monotone_health_prior,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results" / "ppx_v11_complete_structure_ablation"
CONTINUOUS = ROOT / "results" / "ppx_v12_continuous_portfolio_development"
OUT = ROOT / "results" / "structural_prior_set_development"
NAMES = ("affine", "monotone", "history")
SETTINGS = {
    "epsilon": 0.02,
    "cvar_fraction": 0.20,
    "minimum_groups_per_shell": 2,
    "disagreement_quantile": 0.90,
    "ceiling_quantile": 0.99,
    "minimum_relative_gain": 0.0,
    "nested": True,
}


def rate_column(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[:, 1], dtype=np.float64)


def build_priors(train, validation, test):
    affine = fit_affine_prior(train["x"], train["y"], train["groups"])
    monotone = fit_monotone_health_prior(train["health"], train["y"])
    history = fit_history_rate_prior(
        train["health"], rate_column(train["x"]), train["y"], train["groups"]
    )
    models = {"affine": affine, "monotone": monotone, "history": history}
    arrays = {}
    for split, rows in (
        ("validation", validation),
        ("test", test),
    ):
        arrays[split] = np.vstack((
            predict_affine_prior(affine, rows["x"]),
            predict_monotone_health_prior(monotone, rows["health"]),
            predict_history_rate_prior(
                history, rows["health"], rate_column(rows["x"])
            ),
        ))
    return models, arrays


def build_distances(train, validation, test):
    affine_support = fit_support_scale(train["x"])
    history_support = fit_support_scale(np.column_stack([
        train["health"], rate_column(train["x"])
    ]))
    health_scale = max(float(np.std(train["health"])), 1e-12)
    health_min = float(np.min(train["health"]))
    distances = {}
    for split, rows in (("validation", validation), ("test", test)):
        distances[split] = np.vstack((
            support_distance(rows["x"], affine_support),
            health_distance(rows["health"], health_min, health_scale),
            support_distance(
                np.column_stack([rows["health"], rate_column(rows["x"])]),
                history_support,
            ),
        ))
    return distances


def evaluate(name, prepared, source, predictions, continuous_predictions):
    train, validation, test, _, _ = prepared
    fallback_all = expert_portfolios(
        name, source["cohorts"][name], predictions, "validation"
    )
    test_all = expert_portfolios(
        name, source["cohorts"][name], predictions, "test"
    )
    validation_fallback, test_fallback = fallback_all[0], test_all[0]
    _, expert_arrays = build_priors(train, validation, test)
    distances = build_distances(train, validation, test)
    validation_experts = expert_arrays["validation"]
    test_experts = expert_arrays["test"]

    validation_losses = [
        float(np.mean(group_mse(
            validation["y"], prediction, validation["groups"]
        )))
        for prediction in validation_experts
    ]
    best_index = int(np.argmin(validation_losses))
    arms = {
        "fallback": test_fallback,
        "affine": test_experts[0],
        "monotone": test_experts[1],
        "history": test_experts[2],
        "mean_ensemble": np.mean(test_experts, axis=0),
        "best_single_validation": test_experts[best_index],
        "continuous_portfolio": continuous_predictions[name],
    }
    decisions = {}
    for arm, minimum_set_size in (
        ("consensus_only", 2),
        ("nested_approved_set", 1),
    ):
        result = crossfit_structural_prior_set(
            validation["y"],
            validation["groups"],
            validation_fallback,
            validation_experts,
            distances["validation"],
            names=NAMES,
            minimum_set_size=minimum_set_size,
            **SETTINGS,
        )
        prediction, active, width = apply_structural_prior_set(
            test_experts, test_fallback, distances["test"], result.policy
        )
        arms[arm] = prediction
        decisions[arm] = {
            **asdict(result),
            "oof_prediction": None,
            "test_active_fraction": float(np.mean(active)),
            "finite_width_fraction": float(np.mean(np.isfinite(width))),
        }
    return {
        "candidate_priors": NAMES,
        "best_single_index": best_index,
        "best_single_name": NAMES[best_index],
        "validation_group_mse": {
            name: loss for name, loss in zip(NAMES, validation_losses)
        },
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
        raise RuntimeError("refusing to overwrite structural prior-set experiment")
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
        "model": "heterogeneous structural prior-set consensus",
        "protocol": "protocols/STRUCTURAL_PRIOR_SET_DEVELOPMENT_PROTOCOL.md",
        "settings": SETTINGS,
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
