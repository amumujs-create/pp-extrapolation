#!/usr/bin/env python3
"""Exploratory post-reveal CCMR v2.2 adaptation on frozen DS03."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.ds03_prospective import (
    TRAIN_UNITS,
    VALIDATION_UNITS,
    _predict_direct,
    causal_features,
    load_development,
    load_revealed_test,
)
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.small_cohort_route import select_ccmr_v22_route
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

OUT = ROOT / "results/ncmapss_ds03_ccmr_v22"
SELECTION = ROOT / "results/ncmapss_ds03_ppx_v1/selection.json"


def subset(rows, units):
    mask = np.isin(rows["groups"], np.asarray(units, str))
    return {key: value[mask] for key, value in rows.items()}


def anchor_predictions(rows, fits):
    return np.mean([_predict_direct(fit, rows["x"]) for fit in fits], axis=0)


def ccmr_features(rows, anchor):
    correction, context = [], []
    for group in np.unique(rows["groups"]):
        indices = np.flatnonzero(rows["groups"] == group)
        values = anchor[indices]
        cycles = rows["cycles"][indices]
        delta = np.r_[0.0, np.diff(values)]
        acceleration = np.r_[0.0, np.diff(delta)]
        progress = (cycles - cycles.min()) / max(float(np.ptp(cycles)), 1.0)
        correction.append(np.column_stack((delta, acceleration, progress)))
        context.append(np.column_stack((
            values,
            progress,
            delta,
            acceleration,
            rows["x"][indices, :3],
        )))
    # Unit rows are already contiguous and sorted in the adapter.
    return np.concatenate(correction), np.concatenate(context)


def risk_metrics(y, groups, anchor, prediction):
    model = regression_metrics(y, prediction, groups)
    baseline = regression_metrics(y, anchor, groups)
    model_macro = np.mean([row["rmse"] for row in model["per_unit"].values()])
    base_macro = np.mean([row["rmse"] for row in baseline["per_unit"].values()])
    risk = regret_summary(raw_unit_regret(y, groups, anchor, prediction))
    return {
        "pooled_improvement": (
            baseline["pooled"]["rmse"] - model["pooled"]["rmse"]
        ) / baseline["pooled"]["rmse"],
        "macro_improvement": (base_macro - model_macro) / base_macro,
        "raw_regret": {"mean": risk[0], "cvar20": risk[1], "maximum": risk[2]},
        "model": model,
        "anchor": baseline,
    }


def main():
    selection = json.loads(SELECTION.read_text())
    bundle = torch.load(
        SELECTION.with_name(selection["model_artifact"]),
        map_location="cpu",
        weights_only=False,
    )
    fits = bundle["fits"]["direct_fallback"]
    dev = causal_features(
        load_development(ROOT / "data/N-CMAPSS_DS03-012.h5"), "direct"
    )
    test = causal_features(
        load_revealed_test(ROOT / "data/N-CMAPSS_DS03-012.h5"), "direct"
    )
    train = subset(dev, TRAIN_UNITS)
    validation = subset(dev, VALIDATION_UNITS)
    train_anchor = anchor_predictions(train, fits)
    validation_anchor = anchor_predictions(validation, fits)
    test_anchor = anchor_predictions(test, fits)
    train_correction, train_context = ccmr_features(train, train_anchor)
    val_correction, val_context = ccmr_features(validation, validation_anchor)
    test_correction, test_context = ccmr_features(test, test_anchor)

    model = fit_causal_dynamics_bank(
        train_correction,
        train_context,
        train["y"],
        train["groups"],
        train_anchor,
        val_correction,
        val_context,
        validation["y"],
        validation["groups"],
        validation_anchor,
    )
    val_candidate, val_evidence = predict_causal_dynamics_bank(
        model, val_correction, val_context, validation_anchor
    )
    val_metrics = risk_metrics(
        validation["y"], validation["groups"], validation_anchor, val_candidate
    )
    route = select_ccmr_v22_route(
        model,
        len(np.unique(validation["groups"])),
        float(np.mean(val_evidence["active"])),
        val_metrics,
    )
    test_candidate, test_evidence = predict_causal_dynamics_bank(
        model, test_correction, test_context, test_anchor
    )
    deployed = (
        test_candidate
        if route.route in ("stable_bank", "small_crossfit_bank", "cautious_causal")
        else test_anchor
    )
    result = {
        "status": "post-reveal exploratory CCMR v2.2 DS03 adaptation",
        "confirmatory": False,
        "selection_uses_test_outcome": False,
        "anchor": "prospectively selected PP-X direct fallback",
        "route": asdict(route),
        "validation": val_metrics,
        "test_active_fraction": float(np.mean(test_evidence["active"])),
        "test": risk_metrics(test["y"], test["groups"], test_anchor, deployed),
        "test_pooled_r2": regression_metrics(
            test["y"], deployed, test["groups"]
        )["pooled"]["r2"],
        "anchor_pooled_r2": regression_metrics(
            test["y"], test_anchor, test["groups"]
        )["pooled"]["r2"],
        "n_test_units": len(np.unique(test["groups"])),
        "n_test_rows": len(test["y"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        anchor=test_anchor,
        candidate=test_candidate,
        prediction=deployed,
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "route": result["route"],
        "CCMR_v22_r2": result["test_pooled_r2"],
        "anchor_r2": result["anchor_pooled_r2"],
        "test_metrics": result["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
