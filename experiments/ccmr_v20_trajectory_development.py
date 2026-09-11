#!/usr/bin/env python3
"""Non-holdout trajectory development for the CCMR v2.0 predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import ccmr_v17_luminosity_development as lum_dev
import concrete_material_shift_ccmr_v17 as concrete
import lgm50t_expt1_ccmr_v17 as lg
import luminosity_temperature_shift_ccmr_v161 as lum
import radar_nmc_cyclic_ccmr_v18 as radar
import sit_lfp_sequential_ccmr_v17 as sit
from pp_extrapolation.causal_backtest import (
    apply_adaptive_causal_backtest_gate,
)
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import (
    raw_unit_regret,
    regret_summary,
)

OUT = ROOT / "results/ccmr_v20_trajectory_development"
INTERVALS = {
    "Concrete": ((0.0, 0.30), (0.40, 0.60), (0.75, 0.90)),
    "LG_M50T": ((0.0, 0.30), (0.40, 0.60), (0.75, 1.00)),
    "SIT_LFP": ((0.0, 0.30), (0.40, 0.60), (0.75, 0.90)),
    "RADAR_NMC": ((0.0, 0.30), (0.40, 0.60), (0.75, 0.90)),
    "Luminosity": ((0.0, 0.30), (0.40, 0.60), (0.75, 1.00)),
}
V1_ARTIFACTS = {
    "Concrete": (
        "results/concrete_material_shift_ccmr_v17/sealed_predictions.npz"
    ),
    "LG_M50T": (
        "results/lgm50t_expt1_ccmr_v17/sealed_predictions.npz"
    ),
    "SIT_LFP": (
        "results/sit_lfp_sequential_ccmr_v17_amended/"
        "sealed_predictions.npz"
    ),
    "RADAR_NMC": (
        "results/radar_nmc_cyclic_ccmr_v18/sealed_predictions.npz"
    ),
    "Luminosity": (
        "results/luminosity_temperature_shift_ccmr_v161/"
        "sealed_predictions.npz"
    ),
}


def selected(rows, interval):
    return (
        (rows["progress"] >= interval[0])
        & (rows["progress"] <= interval[1])
    )


def subset(rows, choose):
    return {key: value[choose] for key, value in rows.items()}


def trajectory_parts(module):
    trajectories = module.load_trajectories()
    if isinstance(trajectories, tuple):
        trajectories = trajectories[0]
    split = module.split_units(trajectories)
    intervals = INTERVALS[
        "RADAR_NMC" if module is radar
        else "SIT_LFP" if module is sit
        else "LG_M50T"
    ]
    train = module.make_rows(
        trajectories, split["train"], intervals[0]
    )
    validation = module.make_rows(
        trajectories, split["validation"]
    )
    test = module.make_rows(trajectories, split["test"])
    return train, validation, test, intervals


def load_development():
    concrete_trajectories = {
        mode: concrete.load_trajectories(
            concrete.SPLIT_FILES[mode]
        )
        for mode in ("train", "validation", "test")
    }
    concrete_intervals = INTERVALS["Concrete"]
    datasets = {
        "Concrete": (
            concrete.make_rows(
                concrete_trajectories["train"],
                concrete_intervals[0],
            ),
            concrete.make_rows(
                concrete_trajectories["validation"]
            ),
            concrete.make_rows(concrete_trajectories["test"]),
            concrete_intervals,
        ),
        "LG_M50T": trajectory_parts(lg),
        "SIT_LFP": trajectory_parts(sit),
        "RADAR_NMC": trajectory_parts(radar),
    }
    frame = next(iter(pyreadr.read_r(str(lum.DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    temperatures = sorted(frame["celsius"].astype(float).unique())
    split = {
        "train": temperatures[2:],
        "validation": temperatures[1:2],
        "test": temperatures[:1],
    }
    intervals = INTERVALS["Luminosity"]
    datasets["Luminosity"] = (
        lum.make_rows(frame, split["train"], "train"),
        lum_dev.all_rows(frame, split["validation"]),
        lum_dev.all_rows(frame, split["test"]),
        intervals,
    )
    return datasets


def score(rows, prediction, choose):
    y = rows["y"][choose]
    groups = rows["groups"][choose]
    anchor = rows["context"][choose, 0]
    prediction = prediction[choose]
    model = regression_metrics(y, prediction, groups)
    baseline = regression_metrics(y, anchor, groups)
    model_macro = float(np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ]))
    baseline_macro = float(np.mean([
        value["rmse"] for value in baseline["per_unit"].values()
    ]))
    raw = regret_summary(raw_unit_regret(
        y, groups, anchor, prediction
    ))
    return {
        "model": model,
        "persistence": baseline,
        "pooled_improvement": float(
            (baseline["pooled"]["rmse"] - model["pooled"]["rmse"])
            / baseline["pooled"]["rmse"]
        ),
        "macro_improvement": float(
            (baseline_macro - model_macro) / baseline_macro
        ),
        "raw_regret": {
            "mean": raw[0],
            "cvar20": raw[1],
            "maximum": raw[2],
        },
    }


def causal_prediction(candidate, rows, choose):
    return apply_adaptive_causal_backtest_gate(
        candidate,
        rows["context"][:, 0],
        rows["y"],
        rows["groups"],
        rows["origin"],
        rows["target"],
        choose,
        minimum_history=2,
        maximum_history=5,
    )


def v1_score(name, test):
    artifact = np.load(ROOT / V1_ARTIFACTS[name])
    prediction_key = (
        "prediction" if "prediction" in artifact.files else "deployed"
    )
    if not np.allclose(artifact["truth"], test["y"]):
        raise RuntimeError(f"{name}: v1 truth alignment failed")
    return score(
        {
            **test,
            "context": np.column_stack([
                artifact["persistence"],
                np.zeros((len(artifact["truth"]), 6)),
            ]),
        },
        artifact[prediction_key],
        np.ones(len(artifact["truth"]), dtype=bool),
    )


def run_one(name, train, validation, test, intervals):
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
    validation_candidate, validation_base = (
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
    cautious_validation_metrics = score(
        validation, validation_gated, validation_choose
    )
    validation_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    stable_certificate = bool(
        len(np.unique(
            validation["groups"][validation_choose]
        )) >= 10
        and model.validation_active_fraction >= 0.50
        and model.validation_macro_improvement >= 0.05
        and model.validation_mean_regret <= 0.0
        and model.validation_cvar_regret <= 0.01
        and model.validation_max_regret <= 0.02
        and validation_coverage >= 0.10
    )
    cautious_approved = bool(
        validation_coverage >= 0.10
        and cautious_validation_metrics["pooled_improvement"] >= 0.005
        and cautious_validation_metrics["macro_improvement"] >= 0.005
        and cautious_validation_metrics["raw_regret"]["mean"] <= 0.0
        and cautious_validation_metrics["raw_regret"]["cvar20"] <= 0.01
        and cautious_validation_metrics["raw_regret"]["maximum"] <= 0.02
    )
    validation_approved = stable_certificate or cautious_approved
    validation_metrics = (
        score(validation, validation_candidate, validation_choose)
        if stable_certificate else cautious_validation_metrics
    )
    test_candidate, test_base = predict_causal_dynamics_bank(
        model,
        test["correction"],
        test["context"],
        test["context"][:, 0],
    )
    test_gated, test_causal = causal_prediction(
        test_candidate, test, test_choose
    )
    deployed = (
        test_candidate
        if stable_certificate
        else test_gated
        if cautious_approved
        else test["context"][:, 0].copy()
    )
    test_metrics = score(test, deployed, test_choose)
    selected_anchor = test["context"][test_choose, 0]
    selected_prediction = deployed[test_choose]
    fallback = selected_prediction == selected_anchor
    fallback_error = (
        float(np.max(np.abs(
            selected_prediction[fallback] - selected_anchor[fallback]
        )))
        if np.any(fallback) else 0.0
    )
    compact_model = {
        "expert_names": [expert.name for expert in model.experts],
        "expert_weights": model.expert_weights.tolist(),
        "selected_candidate": model.selected_candidate,
        "ridge_alpha": model.ridge_alpha,
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
    }
    return {
        "rows": {
            "train": len(train["y"]),
            "validation": int(np.sum(validation_choose)),
            "test": int(np.sum(test_choose)),
        },
        "units": {
            "train": len(np.unique(train["groups"])),
            "validation": len(np.unique(
                validation["groups"][validation_choose]
            )),
            "test": len(np.unique(test["groups"][test_choose])),
        },
        "model": compact_model,
        "validation": {
            "approved": validation_approved,
            "route": (
                "stable_bank"
                if stable_certificate else "cautious_causal"
                if cautious_approved else "exact_fallback"
            ),
            "causal_coverage": validation_coverage,
            "base_coverage": float(np.mean(
                validation_base["active"][validation_choose]
            )),
            **validation_metrics,
        },
        "test": {
            "deployed_coverage": (
                float(np.mean(test_base["active"][test_choose]))
                if stable_certificate
                else
                float(np.mean(test_causal["active"][test_choose]))
                if cautious_approved else 0.0
            ),
            "base_coverage": float(np.mean(
                test_base["active"][test_choose]
            )),
            "fallback_error": fallback_error,
            **test_metrics,
        },
        "v1_deployed": v1_score(
            name, subset(test, test_choose)
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite v2.0 development")
    datasets = load_development()
    results = {}
    for name, parts in datasets.items():
        results[name] = run_one(name, *parts)
        print(name, flush=True)
    approvals = [
        value["validation"]["approved"] for value in results.values()
    ]
    maximum_regrets = [
        value["test"]["raw_regret"]["maximum"]
        for value in results.values()
    ]
    false_accepts = sum(
        value["validation"]["approved"]
        and value["test"]["pooled_improvement"] < 0
        for value in results.values()
    )
    payload = {
        "status": "non-holdout CCMR v2.0 trajectory development",
        "holdouts_loaded": False,
        "cohorts": results,
        "leave_one_cohort_out_summary": {
            "cohorts": len(results),
            "validation_approvals": int(sum(approvals)),
            "false_accepts": int(false_accepts),
            "maximum_test_raw_regret": float(max(maximum_regrets)),
            "exact_fallback_all": bool(all(
                value["test"]["fallback_error"] == 0
                for value in results.values()
            )),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["leave_one_cohort_out_summary"], indent=2))


if __name__ == "__main__":
    main()
