#!/usr/bin/env python3
"""Retrospective robustness audit for Stability-First RBPR v1.3."""
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
from pp_extrapolation.risk_budgeted_prior import apply_prior_residual
from pp_extrapolation.stability_first import (
    regret_summary,
    select_stability_first_prior,
    unit_regret,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
FIXED = ROOT / "results/risk_budgeted_prior_residual_development/results.json"
CONTINUOUS = ROOT / "results/ppx_v12_continuous_portfolio_development/results.json"
OUT = ROOT / "results/stability_first_rbpr_v13_development"
TRUSTS = np.array([0.02, 0.05, 0.10, 0.20, 0.40])


def replay(name, prepared, cohort, predictions, fixed, continuous):
    _, validation, test, _, _ = prepared
    val_baseline, val_priors, _ = portfolios(
        name, cohort, predictions, "validation"
    )
    test_baseline, test_priors, _ = portfolios(
        name, cohort, predictions, "test"
    )
    val_experts = np.stack([val_priors[float(trust)] for trust in TRUSTS])
    test_experts = np.stack([test_priors[float(trust)] for trust in TRUSTS])
    decision = select_stability_first_prior(
        validation["y"], validation["groups"], val_baseline,
        val_experts, TRUSTS,
    )
    if decision.accepted:
        index = int(np.flatnonzero(np.isclose(
            TRUSTS, decision.selected_trust
        ))[0])
        validation_prediction = apply_prior_residual(
            val_baseline, val_experts[index], decision.alpha
        )
        test_prediction = apply_prior_residual(
            test_baseline, test_experts[index], decision.alpha
        )
    else:
        validation_prediction = val_baseline.copy()
        test_prediction = test_baseline.copy()
    test_regret = regret_summary(unit_regret(
        test["y"], test["groups"], test_baseline, test_prediction
    ))
    return {
        "decision": asdict(decision),
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
        "fallback_max_abs_error": float(np.max(np.abs(
            test_prediction - test_baseline
        ))) if not decision.accepted else None,
        "comparators": {
            "baseline": regression_metrics(
                test["y"], test_baseline, test["groups"]
            ),
            "fixed_rbpr": fixed["cohorts"][name]["arms"]["local_cvar"]["test"],
            "continuous_portfolio": continuous["cohorts"][name][
                "portfolio_test"
            ],
        },
    }, test_prediction


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite stability-first replay")
    source = json.loads((SOURCE / "results.json").read_text())
    fixed = json.loads(FIXED.read_text())
    continuous = json.loads(CONTINUOUS.read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        results[name], arrays[name] = replay(
            name, prepared, source["cohorts"][name], predictions,
            fixed, continuous,
        )
        print(name, json.dumps({
            "decision": results[name]["decision"],
            "test": results[name]["test"]["pooled"],
            "test_regret": results[name]["test_regret"],
        }), flush=True)
    target.write_text(json.dumps({
        "status": "post-test stability-first development; not confirmation",
        "model": "Stability-First RBPR v1.3",
        "protocol": "protocols/STABILITY_FIRST_RBPR_V13_PROTOCOL.md",
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
