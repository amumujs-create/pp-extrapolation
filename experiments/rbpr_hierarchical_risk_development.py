#!/usr/bin/env python3
"""Hierarchical small-unit risk estimator for monotone RBPR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from rbpr_crossfit_monotone_development import fit_mode
from risk_budgeted_prior_residual_development import positive_scale, portfolios
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.risk_budgeted_prior import (
    fit_support_scale,
    support_distance,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
FIXED = ROOT / "results/risk_budgeted_prior_residual_development/results.json"
RAW = ROOT / "results/rbpr_crossfit_monotone_development/results.json"
OUT = ROOT / "results/rbpr_hierarchical_risk_development"
TRUSTS = (0.02, 0.05, 0.10, 0.20, 0.40)
PRIMARY_N0 = 5.0


def replay(name, prepared, cohort, predictions, fixed, raw):
    train, validation, test, _, _ = prepared
    val_baseline, val_priors, val_disagreement = portfolios(
        name, cohort, predictions, "validation"
    )
    test_baseline, test_priors, test_disagreement = portfolios(
        name, cohort, predictions, "test"
    )
    support = fit_support_scale(train["x"])
    val_distance = support_distance(validation["x"], support)
    test_distance = support_distance(test["x"], support)
    distance_scale = positive_scale(val_distance)
    val_distance /= distance_scale
    test_distance /= distance_scale
    disagreement_scales = {
        trust: positive_scale(val_disagreement[trust]) for trust in TRUSTS
    }
    val_disagreement = {
        trust: val_disagreement[trust] / disagreement_scales[trust]
        for trust in TRUSTS
    }
    test_disagreement = {
        trust: test_disagreement[trust] / disagreement_scales[trust]
        for trust in TRUSTS
    }
    baseline = {"validation": val_baseline, "test": test_baseline}
    priors = {"validation": val_priors, "test": test_priors}

    arms, arrays = {}, {}
    for mode in ("global", "distance_only", "disagreement_only", "full"):
        arms[mode], arrays[mode] = fit_mode(
            validation, test, baseline, priors, val_distance, test_distance,
            val_disagreement, test_disagreement, mode,
            shrinkage_n0=PRIMARY_N0,
        )
    sensitivity = {}
    for n0 in (0.0, 2.0, 10.0):
        label = f"n0_{n0:g}"
        sensitivity[label], arrays[label] = fit_mode(
            validation, test, baseline, priors, val_distance, test_distance,
            val_disagreement, test_disagreement, "full",
            shrinkage_n0=n0,
        )
    arms["baseline"] = {
        "test": regression_metrics(test["y"], test_baseline, test["groups"])
    }
    arms["fixed_rbpr"] = fixed["cohorts"][name]["arms"]["local_cvar"]
    arms["raw_crossfit"] = raw["cohorts"][name]["arms"]["full"]
    return {
        "primary_shrinkage_n0": PRIMARY_N0,
        "distance_scale": distance_scale,
        "disagreement_scales": disagreement_scales,
        "arms": arms,
        "shrinkage_sensitivity": sensitivity,
    }, arrays


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite hierarchical-risk replay")
    source = json.loads((SOURCE / "results.json").read_text())
    fixed = json.loads(FIXED.read_text())
    raw = json.loads(RAW.read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        results[name], cohort_arrays = replay(
            name, prepared, source["cohorts"][name], predictions, fixed, raw
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
        "status": "post-test hierarchical-risk development; not confirmation",
        "protocol": "protocols/RBPR_HIERARCHICAL_RISK_DEVELOPMENT_PROTOCOL.md",
        "primary_shrinkage_n0": PRIMARY_N0,
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
