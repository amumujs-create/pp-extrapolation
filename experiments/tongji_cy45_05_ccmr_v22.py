#!/usr/bin/env python3
"""One-shot CCMR v2.2 on Tongji CY45-05."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import tongji_cy45_05_ccmr_v20 as base
from ccmr_v20_trajectory_development import causal_prediction, score
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.small_cohort_route import select_ccmr_v22_route

OUT = ROOT / "results/tongji_cy45_05_ccmr_v22"
MANIFEST = ROOT / "protocols/CCMR_V22_FROZEN_MANIFEST.json"
PROTOCOL = ROOT / "protocols/TONGJI_CY45_05_CCMR_V22_PROTOCOL.md"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite Tongji CY45-05 CCMR v2.2")
    trajectories = base.load_trajectories()
    split = base.split_units(trajectories)
    train = base.make_rows(
        trajectories, split["train"], base.TRAIN_INTERVAL
    )
    validation = base.make_rows(trajectories, split["validation"])
    test = base.make_rows(trajectories, split["test"])
    validation_choose = base.select(validation, base.VAL_INTERVAL)
    test_choose = base.select(test, base.TEST_INTERVAL)
    validation_fit = {
        key: value[validation_choose]
        if isinstance(value, np.ndarray) else value
        for key, value in validation.items()
    }
    admissible = bool(
        len(trajectories) >= 40
        and np.sum(validation_choose) >= 20
        and np.sum(test_choose) >= 15
        and len(train["y"]) > 0
        and np.min(test["progress"][test_choose]) > np.max(train["progress"])
    )
    payload = {
        "status": "inconclusive before model fitting",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "predictor_manifest": str(MANIFEST.relative_to(ROOT)),
        "predictor_manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "eligible_cells": len(trajectories),
        "split": {key: list(value) for key, value in split.items()},
        "validation_origins": int(np.sum(validation_choose)),
        "test_origins": int(np.sum(test_choose)),
        "confirmatory": False,
        "dataset_level_untouched": False,
        "ccmr_contract_new": True,
        "ccmr_version": "v2.2",
        "counted_sealed_auto_promote": False,
    }
    if not admissible:
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        return

    model = fit_causal_dynamics_bank(
        train["correction"], train["context"], train["y"],
        train["groups"], train["context"][:, 0],
        validation_fit["correction"], validation_fit["context"],
        validation_fit["y"], validation_fit["groups"],
        validation_fit["context"][:, 0],
    )
    validation_candidate, validation_base = predict_causal_dynamics_bank(
        model, validation["correction"], validation["context"],
        validation["context"][:, 0],
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    validation_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    validation_units = len(np.unique(
        validation["groups"][validation_choose]
    ))
    cautious_metrics = score(
        validation, validation_gated, validation_choose
    )
    route = select_ccmr_v22_route(
        model, validation_units, validation_coverage, cautious_metrics
    )
    test_candidate, test_base = predict_causal_dynamics_bank(
        model, test["correction"], test["context"], test["context"][:, 0],
    )
    test_gated, test_causal = causal_prediction(
        test_candidate, test, test_choose
    )
    if route.route in ("stable_bank", "small_crossfit_bank"):
        deployed = test_candidate
        coverage = float(np.mean(test_base["active"][test_choose]))
    elif route.route == "cautious_causal":
        deployed = test_gated
        coverage = float(np.mean(test_causal["active"][test_choose]))
    else:
        deployed = test["context"][:, 0].copy()
        coverage = 0.0
    selected_anchor = test["context"][test_choose, 0]
    selected_prediction = deployed[test_choose]
    inactive = selected_prediction == selected_anchor
    fallback_error = (
        float(np.max(np.abs(
            selected_prediction[inactive] - selected_anchor[inactive]
        )))
        if np.any(inactive) else 0.0
    )
    test_metrics = score(test, deployed, test_choose)
    success = bool(
        float(np.min(test["progress"][test_choose]))
        > float(np.max(train["progress"]))
        and test_metrics["model"]["pooled"]["r2"] > 0.0
        and test_metrics["pooled_improvement"] >= 0.005
        and test_metrics["macro_improvement"] >= 0.005
        and test_metrics["raw_regret"]["mean"] <= 0.0
        and test_metrics["raw_regret"]["cvar20"] <= 0.01
        and test_metrics["raw_regret"]["maximum"] <= 0.02
        and coverage >= 0.10
        and fallback_error == 0.0
        and route.approved
    )
    payload.update({
        "status": "performance success" if success else "safe fallback or failure",
        "performance_success": success,
        "route": asdict(route),
        "validation_units": validation_units,
        "model": {
            "selected_candidate": model.selected_candidate,
            "expert_names": [expert.name for expert in model.experts],
            "expert_weights": model.expert_weights.tolist(),
            "ridge_alpha": model.ridge_alpha,
            "deployment_mass": model.deployment_mass,
            "validation_macro_improvement": model.validation_macro_improvement,
            "validation_active_fraction": model.validation_active_fraction,
        },
        "validation": {
            "causal_shadow_coverage": validation_coverage,
            "metrics": (
                score(validation, validation_candidate, validation_choose)
                if route.route in ("stable_bank", "small_crossfit_bank")
                else cautious_metrics
            ),
        },
        "test": {
            "deployed_coverage": coverage,
            "fallback_error": fallback_error,
            **test_metrics,
        },
    })
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        candidate=test_candidate[test_choose],
        deployed=selected_prediction,
        persistence=selected_anchor,
        truth=test["y"][test_choose],
        groups=test["groups"][test_choose].astype(str),
        origin=test["origin"][test_choose],
        target=test["target"][test_choose],
        progress=test["progress"][test_choose],
        route=np.asarray([route.route]),
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
