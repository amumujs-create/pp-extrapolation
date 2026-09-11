#!/usr/bin/env python3
"""Cross-fitted monotone Risk-Budgeted Prior Residual ablation."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from risk_budgeted_prior_residual_development import (
    positive_scale,
    portfolios,
)
from pp_extrapolation.crossfit_monotone import (
    crossfit_monotone_budget,
    hierarchical_risk_ratios,
    monotone_modulation,
    risk_ratios,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.risk_budgeted_prior import (
    apply_prior_residual,
    fit_support_scale,
    group_mse,
    support_distance,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
FIXED = ROOT / "results/risk_budgeted_prior_residual_development/results.json"
OUT = ROOT / "results/rbpr_crossfit_monotone_development"
TRUSTS = (0.02, 0.05, 0.10, 0.20, 0.40)
EPSILON = 0.02


def fit_mode(
    validation, test, baseline, priors, val_distance, test_distance,
    val_disagreement, test_disagreement, mode, shrinkage_n0=0.0,
):
    settings = {
        "epsilon": EPSILON,
        "use_distance": mode in ("distance_only", "full"),
        "use_disagreement": mode in ("disagreement_only", "full"),
        "shrinkage_n0": shrinkage_n0,
    }
    candidates = []
    for trust in TRUSTS:
        result = crossfit_monotone_budget(
            validation["y"], validation["groups"], baseline["validation"],
            priors["validation"][trust], val_distance,
            val_disagreement[trust], **settings,
        )
        oof_loss = float(np.mean(group_mse(
            validation["y"], result.oof_prediction, validation["groups"]
        )))
        candidates.append((not result.oof_feasible, oof_loss, trust, result))
    feasible_failure, _, trust, result = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    if feasible_failure:
        trust = 0.0
        test_prediction = baseline["test"].copy()
        validation_prediction = baseline["validation"].copy()
        alpha = distance_decay = disagreement_decay = 0.0
    else:
        fit = result.final
        alpha = fit.alpha
        distance_decay = fit.distance_decay
        disagreement_decay = fit.disagreement_decay
        validation_modulation = monotone_modulation(
            val_distance, val_disagreement[trust], distance_decay,
            disagreement_decay,
        )
        test_modulation = monotone_modulation(
            test_distance, test_disagreement[trust], distance_decay,
            disagreement_decay,
        )
        validation_prediction = apply_prior_residual(
            baseline["validation"], priors["validation"][trust], alpha,
            validation_modulation,
        )
        test_prediction = apply_prior_residual(
            baseline["test"], priors["test"][trust], alpha, test_modulation
        )
    test_mean, test_tail = risk_ratios(
        test["y"], test["groups"], baseline["test"], test_prediction
    )
    _, test_hierarchical_tail = hierarchical_risk_ratios(
        test["y"], test["groups"], baseline["test"], test_prediction,
        shrinkage_n0=shrinkage_n0,
    )
    return {
        "selected_trust": trust,
        "alpha": alpha,
        "distance_decay": distance_decay,
        "disagreement_decay": disagreement_decay,
        "oof_feasible": not feasible_failure,
        "oof_mean_excess_ratio": (
            result.oof_mean_excess_ratio if not feasible_failure else 0.0
        ),
        "oof_tail_excess_ratio": (
            result.oof_tail_excess_ratio if not feasible_failure else 0.0
        ),
        "final_fit": asdict(result.final) if not feasible_failure else None,
        "fold_parameters": (
            result.fold_parameters if not feasible_failure else ()
        ),
        "validation": regression_metrics(
            validation["y"], validation_prediction, validation["groups"]
        ),
        "test": regression_metrics(
            test["y"], test_prediction, test["groups"]
        ),
        "test_mean_excess_ratio": test_mean,
        "test_tail_excess_ratio": test_tail,
        "test_hierarchical_tail_excess_ratio": test_hierarchical_tail,
    }, test_prediction


def replay(name, prepared, cohort, predictions, fixed):
    train, validation, test, _, _ = prepared
    val_baseline, val_priors, val_disagreement = portfolios(
        name, cohort, predictions, "validation"
    )
    test_baseline, test_priors, test_disagreement = portfolios(
        name, cohort, predictions, "test"
    )
    support = fit_support_scale(train["x"])
    val_distance_raw = support_distance(validation["x"], support)
    test_distance_raw = support_distance(test["x"], support)
    distance_scale = positive_scale(val_distance_raw)
    val_distance = val_distance_raw / distance_scale
    test_distance = test_distance_raw / distance_scale
    disagreement_scales = {
        trust: positive_scale(val_disagreement[trust]) for trust in TRUSTS
    }
    normalized_val_disagreement = {
        trust: val_disagreement[trust] / disagreement_scales[trust]
        for trust in TRUSTS
    }
    normalized_test_disagreement = {
        trust: test_disagreement[trust] / disagreement_scales[trust]
        for trust in TRUSTS
    }
    baseline = {"validation": val_baseline, "test": test_baseline}
    priors = {"validation": val_priors, "test": test_priors}
    arms, arrays = {}, {}
    for mode in ("global", "distance_only", "disagreement_only", "full"):
        arms[mode], arrays[mode] = fit_mode(
            validation, test, baseline, priors, val_distance, test_distance,
            normalized_val_disagreement, normalized_test_disagreement, mode,
        )
    arms["baseline"] = {
        "test": regression_metrics(test["y"], test_baseline, test["groups"])
    }
    arms["original_fixed_rbpr"] = fixed["cohorts"][name]["arms"]["local_cvar"]
    return {
        "distance_scale": distance_scale,
        "disagreement_scales": disagreement_scales,
        "arms": arms,
    }, arrays


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite crossfit monotone replay")
    source = json.loads((SOURCE / "results.json").read_text())
    fixed = json.loads(FIXED.read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        results[name], cohort_arrays = replay(
            name, prepared, source["cohorts"][name], predictions, fixed
        )
        arrays.update({
            f"{name}_{arm}": prediction
            for arm, prediction in cohort_arrays.items()
        })
        print(name, {
            arm: round(row["test"]["pooled"]["r2"], 4)
            for arm, row in results[name]["arms"].items()
        }, flush=True)
    target.write_text(json.dumps({
        "status": "post-test crossfit-monotone development; not confirmation",
        "protocol": "protocols/RBPR_CROSSFIT_MONOTONE_DEVELOPMENT_PROTOCOL.md",
        "epsilon": EPSILON,
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
