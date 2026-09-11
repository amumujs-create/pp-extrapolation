#!/usr/bin/env python3
"""Frozen retrospective ablation of Risk-Budgeted Prior Residual."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.risk_budgeted_prior import (
    apply_prior_residual,
    fit_support_scale,
    group_mse,
    local_budget_modulation,
    select_risk_budget,
    support_distance,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
CONTINUOUS = ROOT / "results/ppx_v12_continuous_portfolio_development/results.json"
OUT = ROOT / "results/risk_budgeted_prior_residual_development"
TRUSTS = (0.02, 0.05, 0.10, 0.20, 0.40)
EPSILON = 0.02


def route_members(name, cohort, predictions, trust, split):
    rows = [row for row in cohort["grid_rows"] if np.isclose(row["trust"], trust)]
    return np.concatenate([
        predictions[f"{name}_{row['key']}_{split}"] for row in rows
    ], axis=0)


def portfolios(name, cohort, predictions, split):
    baseline_members = route_members(name, cohort, predictions, 0.0, split)
    prior_members = {
        trust: route_members(name, cohort, predictions, trust, split)
        for trust in TRUSTS
    }
    return (
        baseline_members.mean(axis=0),
        {trust: members.mean(axis=0) for trust, members in prior_members.items()},
        {
            trust: np.std(
                prior_members[trust] - baseline_members, axis=0, ddof=1
            )
            for trust in TRUSTS
        },
    )


def positive_scale(values):
    values = np.asarray(values)
    positive = values[values > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def validation_score(y, groups, prediction):
    return float(np.mean(group_mse(y, prediction, groups)))


def choose_arm(
    validation, test, baseline, priors, val_mod, test_mod, mode,
    epsilon=EPSILON,
):
    uses_local = mode in (
        "local_unconstrained", "local_cvar", "support_only_cvar",
        "uncertainty_only_cvar",
    )
    candidates = []
    for trust in TRUSTS:
        if mode == "unconstrained":
            decision = None
            alpha = 1.0
            modulation = None
        elif mode == "local_unconstrained":
            decision = None
            alpha = 1.0
            modulation = val_mod[trust]
        else:
            modulation = val_mod[trust] if uses_local else None
            decision = select_risk_budget(
                validation["y"], validation["groups"], baseline["validation"],
                priors["validation"][trust], modulation,
                epsilon=epsilon, enforce_tail=mode != "global_mean",
            )
            alpha = decision.alpha
        prediction = apply_prior_residual(
            baseline["validation"], priors["validation"][trust],
            alpha, modulation,
        )
        candidates.append((
            validation_score(validation["y"], validation["groups"], prediction),
            alpha, trust, decision,
        ))
    _, alpha, trust, decision = min(candidates, key=lambda row: (
        row[0], row[1], row[2]
    ))
    validation_mod = val_mod[trust] if uses_local else None
    test_modulation = test_mod[trust] if uses_local else None
    val_prediction = apply_prior_residual(
        baseline["validation"], priors["validation"][trust],
        alpha, validation_mod,
    )
    test_prediction = apply_prior_residual(
        baseline["test"], priors["test"][trust], alpha, test_modulation
    )
    return {
        "selected_trust": trust,
        "alpha": alpha,
        "risk_decision": asdict(decision) if decision else None,
        "validation": regression_metrics(
            validation["y"], val_prediction, validation["groups"]
        ),
        "test": regression_metrics(test["y"], test_prediction, test["groups"]),
        "mean_test_modulation": (
            float(np.mean(test_modulation)) if test_modulation is not None else 1.0
        ),
        "mean_effective_test_alpha": (
            float(alpha * np.mean(test_modulation))
            if test_modulation is not None else alpha
        ),
    }, test_prediction


def replay(name, prepared, cohort, predictions, continuous):
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
    val_mod, test_mod = {}, {}
    val_support_mod, test_support_mod = {}, {}
    val_uncertainty_mod, test_uncertainty_mod = {}, {}
    disagreement_scales = {}
    for trust in TRUSTS:
        disagreement_scales[trust] = positive_scale(val_disagreement[trust])
        val_mod[trust] = local_budget_modulation(
            val_distance, val_disagreement[trust], distance_scale,
            disagreement_scales[trust],
        )
        test_mod[trust] = local_budget_modulation(
            test_distance, test_disagreement[trust], distance_scale,
            disagreement_scales[trust],
        )
        val_support_mod[trust] = local_budget_modulation(
            val_distance, np.zeros_like(val_distance), distance_scale, 1.0,
        )
        test_support_mod[trust] = local_budget_modulation(
            test_distance, np.zeros_like(test_distance), distance_scale, 1.0,
        )
        val_uncertainty_mod[trust] = local_budget_modulation(
            np.zeros_like(val_distance), val_disagreement[trust], 1.0,
            disagreement_scales[trust],
        )
        test_uncertainty_mod[trust] = local_budget_modulation(
            np.zeros_like(test_distance), test_disagreement[trust], 1.0,
            disagreement_scales[trust],
        )
    baseline = {"validation": val_baseline, "test": test_baseline}
    priors = {"validation": val_priors, "test": test_priors}
    arms, arrays = {}, {}
    for mode in (
        "unconstrained", "local_unconstrained", "global_mean",
        "global_cvar", "local_cvar",
    ):
        arms[mode], arrays[mode] = choose_arm(
            validation, test, baseline, priors, val_mod, test_mod, mode
        )
    for mode, validation_mod, test_modulation in (
        ("support_only_cvar", val_support_mod, test_support_mod),
        ("uncertainty_only_cvar", val_uncertainty_mod, test_uncertainty_mod),
    ):
        arms[mode], arrays[mode] = choose_arm(
            validation, test, baseline, priors, validation_mod,
            test_modulation, mode,
        )
    sensitivity = {}
    for epsilon in (0.0, 0.01, 0.02, 0.05, 0.10):
        label = f"epsilon_{epsilon:.2f}"
        sensitivity[label], arrays[label] = choose_arm(
            validation, test, baseline, priors, val_mod, test_mod,
            "local_cvar", epsilon=epsilon,
        )
    arms["baseline"] = {
        "validation": regression_metrics(
            validation["y"], val_baseline, validation["groups"]
        ),
        "test": regression_metrics(test["y"], test_baseline, test["groups"]),
    }
    arms["continuous_portfolio"] = {
        "test": continuous["cohorts"][name]["portfolio_test"],
        "decision": continuous["cohorts"][name]["decision"],
    }
    return {
        "support": {
            "validation_distance_scale": distance_scale,
            "validation_distance_mean": float(np.mean(val_distance)),
            "test_distance_mean": float(np.mean(test_distance)),
            "disagreement_scales": disagreement_scales,
        },
        "arms": arms,
        "epsilon_sensitivity": sensitivity,
    }, arrays


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite frozen RBPR development run")
    source = json.loads((SOURCE / "results.json").read_text())
    continuous = json.loads(CONTINUOUS.read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        results[name], cohort_arrays = replay(
            name, prepared, source["cohorts"][name], predictions, continuous
        )
        arrays.update({
            f"{name}_{arm}": prediction
            for arm, prediction in cohort_arrays.items()
        })
        print(name, {
            arm: round(row["test"]["pooled"]["r2"], 4)
            for arm, row in results[name]["arms"].items()
            if "test" in row
        }, flush=True)
    payload = {
        "status": "post-test RBPR method development; not confirmation",
        "frozen_protocol": "protocols/RISK_BUDGETED_PRIOR_RESIDUAL_DEVELOPMENT_PROTOCOL.md",
        "model": "Risk-Budgeted Prior Residual",
        "epsilon": EPSILON,
        "cvar_fraction": 0.20,
        "cohorts": results,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
