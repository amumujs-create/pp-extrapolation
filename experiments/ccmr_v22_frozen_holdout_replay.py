#!/usr/bin/env python3
"""Single frozen holdout replay for promoted CCMR v2.2."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_frozen_holdout_replay import load_holdouts
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
from pp_extrapolation.small_cohort_route import select_ccmr_v22_route

OUT = ROOT / "results/ccmr_v22_frozen_holdout_replay"
MANIFEST = ROOT / "protocols/CCMR_V22_FROZEN_MANIFEST.json"
V20 = ROOT / "results/ccmr_v20_frozen_holdout_replay/results.json"


def evaluate(train, validation, test, intervals):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    model = fit_causal_dynamics_bank(
        train["correction"], train["context"], train["y"],
        train["groups"], train["context"][:, 0],
        validation_fit["correction"], validation_fit["context"],
        validation_fit["y"], validation_fit["groups"],
        validation_fit["context"][:, 0],
    )
    validation_candidate, _ = predict_causal_dynamics_bank(
        model, validation["correction"], validation["context"],
        validation["context"][:, 0],
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    causal_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    cautious_metrics = score(
        validation, validation_gated, validation_choose
    )
    validation_units = len(np.unique(
        validation["groups"][validation_choose]
    ))
    route = select_ccmr_v22_route(
        model, validation_units, causal_coverage, cautious_metrics
    )
    candidate, evidence = predict_causal_dynamics_bank(
        model, test["correction"], test["context"],
        test["context"][:, 0],
    )
    gated, causal = causal_prediction(candidate, test, test_choose)
    if route.route in ("stable_bank", "small_crossfit_bank"):
        deployed = candidate
        coverage = float(np.mean(evidence["active"][test_choose]))
    elif route.route == "cautious_causal":
        deployed = gated
        coverage = float(np.mean(causal["active"][test_choose]))
    else:
        deployed = test["context"][:, 0].copy()
        coverage = 0.0
    prediction = deployed[test_choose]
    anchor = test["context"][test_choose, 0]
    inactive = prediction == anchor
    fallback_error = (
        float(np.max(np.abs(
            prediction[inactive] - anchor[inactive]
        ))) if np.any(inactive) else 0.0
    )
    return {
        "route": asdict(route),
        "validation_units": validation_units,
        "validation_causal_coverage": causal_coverage,
        "test": {
            "coverage": coverage,
            "fallback_error": fallback_error,
            **score(test, deployed, test_choose),
        },
    }, prediction, test["y"][test_choose], test["groups"][test_choose], anchor


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite v2.2 frozen replay")
    old = json.loads(V20.read_text())["cohorts"]
    payload = {
        "status": "frozen CCMR v2.2 holdout replay complete",
        "confirmatory": False,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "cohorts": {},
    }
    arrays = {}
    for name, values in load_holdouts().items():
        train, validation, test, intervals = values[:4]
        result, prediction, truth, groups, anchor = evaluate(
            train, validation, test, intervals
        )
        result["comparisons"] = {
            "CCMR_v20": old[name]["test"],
            "Engression": old[name]["comparisons"]["Engression"],
        }
        payload["cohorts"][name] = result
        arrays[f"{name}_prediction"] = prediction
        arrays[f"{name}_truth"] = truth
        arrays[f"{name}_groups"] = groups.astype(str)
        arrays[f"{name}_persistence"] = anchor
        print(name, flush=True)
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
