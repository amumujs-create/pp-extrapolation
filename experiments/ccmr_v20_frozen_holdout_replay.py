#!/usr/bin/env python3
"""Single replay of frozen CCMR v2.0 on the two excluded holdouts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import alloya_ccmr_v19_posthoc as alloy
import alloya_unopened_ccmr_v16 as alloy_v1
import multistage_rpt_ccmr_v19 as multi
from ccmr_v20_trajectory_development import (
    causal_prediction,
    score,
    selected,
    subset,
)
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)

OUT = ROOT / "results/ccmr_v20_frozen_holdout_replay"
MANIFEST = ROOT / "protocols/CCMR_V20_FROZEN_PREDICTOR_MANIFEST.json"
COMPARATORS = ROOT / (
    "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative"
)


def load_holdouts():
    frame = next(iter(pyreadr.read_r(str(alloy_v1.DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    alloy_split = alloy_v1.frozen_split(
        sorted(frame["specimen"].astype(str).unique())
    )
    alloy_full = {
        mode: alloy.make_all_rows(frame, alloy_split[mode])
        for mode in ("train", "validation", "test")
    }
    alloy_intervals = ((0.0, 0.30), (0.40, 0.60), (0.75, 1.0))
    alloy_train = subset(
        alloy_full["train"],
        selected(alloy_full["train"], alloy_intervals[0]),
    )

    trajectories = multi.load_trajectories()
    multi_split = multi.split_units(trajectories)
    multi_full = {
        mode: multi.make_rows(trajectories, multi_split[mode])
        for mode in ("train", "validation", "test")
    }
    multi_intervals = ((0.0, 0.30), (0.40, 0.60), (0.75, 0.90))
    multi_train = subset(
        multi_full["train"],
        selected(multi_full["train"], multi_intervals[0]),
    )
    return {
        "Alloy_A": (
            alloy_train,
            alloy_full["validation"],
            alloy_full["test"],
            alloy_intervals,
            ROOT / "results/alloya_unopened_ccmr_v16/"
            "sealed_predictions.npz",
            "prediction",
            COMPARATORS / "alloy_a_ensemble_predictions.npz",
        ),
        "MultiStage_RPT": (
            multi_train,
            multi_full["validation"],
            multi_full["test"],
            multi_intervals,
            ROOT / "results/multistage_rpt_ccmr_v19_amended2/"
            "sealed_predictions.npz",
            "deployed",
            COMPARATORS / "multistage_rpt_ensemble_predictions.npz",
        ),
    }


def evaluate(name, train, validation, test, intervals):
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
    validation_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    stable = bool(
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
    cautious_metrics = score(
        validation, validation_gated, validation_choose
    )
    cautious = bool(
        validation_coverage >= 0.10
        and cautious_metrics["pooled_improvement"] >= 0.005
        and cautious_metrics["macro_improvement"] >= 0.005
        and cautious_metrics["raw_regret"]["mean"] <= 0.0
        and cautious_metrics["raw_regret"]["cvar20"] <= 0.01
        and cautious_metrics["raw_regret"]["maximum"] <= 0.02
    )
    route = (
        "stable_bank" if stable
        else "cautious_causal" if cautious
        else "exact_fallback"
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
        test_candidate if stable
        else test_gated if cautious
        else test["context"][:, 0].copy()
    )
    selected_anchor = test["context"][test_choose, 0]
    selected_prediction = deployed[test_choose]
    inactive = selected_prediction == selected_anchor
    fallback_error = (
        float(np.max(np.abs(
            selected_prediction[inactive] - selected_anchor[inactive]
        )))
        if np.any(inactive) else 0.0
    )
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
        "route": route,
        "model": {
            "selected_candidate": model.selected_candidate,
            "expert_names": [expert.name for expert in model.experts],
            "expert_weights": model.expert_weights.tolist(),
            "ridge_alpha": model.ridge_alpha,
            "deployment_mass": model.deployment_mass,
            "validation_macro_improvement": (
                model.validation_macro_improvement
            ),
            "validation_active_fraction": (
                model.validation_active_fraction
            ),
        },
        "validation": {
            "causal_shadow_coverage": validation_coverage,
            "base_coverage": float(np.mean(
                validation_base["active"][validation_choose]
            )),
            "metrics": (
                score(
                    validation,
                    validation_candidate,
                    validation_choose,
                )
                if stable else cautious_metrics
            ),
        },
        "test": {
            "deployed_coverage": (
                float(np.mean(test_base["active"][test_choose]))
                if stable else
                float(np.mean(test_causal["active"][test_choose]))
                if cautious else 0.0
            ),
            "fallback_error": fallback_error,
            **score(test, deployed, test_choose),
        },
        "prediction": selected_prediction,
        "truth": test["y"][test_choose],
        "groups": test["groups"][test_choose],
        "persistence": selected_anchor,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen holdout replay")
    manifest_bytes = MANIFEST.read_bytes()
    result = {
        "status": "frozen CCMR v2.0 holdout replay complete",
        "confirmatory": False,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "cohorts": {},
    }
    predictions = {}
    for name, values in load_holdouts().items():
        (
            train,
            validation,
            test,
            intervals,
            v1_path,
            v1_key,
            comparator_path,
        ) = values
        evaluated = evaluate(
            name, train, validation, test, intervals
        )
        v1 = np.load(v1_path)
        comparators = np.load(comparator_path)
        if not np.allclose(v1["truth"], evaluated["truth"]):
            raise RuntimeError(f"{name}: v1 alignment failure")
        if not np.allclose(
            comparators["truth"], evaluated["truth"]
        ):
            raise RuntimeError(f"{name}: comparator alignment failure")
        result["cohorts"][name] = {
            key: value for key, value in evaluated.items()
            if key not in (
                "prediction", "truth", "groups", "persistence"
            )
        }
        result["cohorts"][name]["comparisons"] = {
            "CCMR_previous": score(
                {
                    **subset(test, selected(test, intervals[2])),
                    "context": np.column_stack([
                        v1["persistence"],
                        np.zeros((len(v1["truth"]), 6)),
                    ]),
                },
                v1[v1_key],
                np.ones(len(v1["truth"]), dtype=bool),
            ),
            "Engression": score(
                {
                    **subset(test, selected(test, intervals[2])),
                    "context": np.column_stack([
                        comparators["persistence"],
                        np.zeros((len(comparators["truth"]), 6)),
                    ]),
                },
                comparators["Engression"],
                np.ones(len(comparators["truth"]), dtype=bool),
            ),
        }
        predictions[f"{name}_truth"] = evaluated["truth"]
        predictions[f"{name}_groups"] = evaluated["groups"].astype(str)
        predictions[f"{name}_persistence"] = evaluated["persistence"]
        predictions[f"{name}_CCMR_v20"] = evaluated["prediction"]
        print(name, flush=True)
    np.savez_compressed(OUT / "predictions.npz", **predictions)
    result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
