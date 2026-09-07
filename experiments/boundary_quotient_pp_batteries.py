#!/usr/bin/env python3
"""Relationship-shift screen for boundary-quotient PP on three batteries."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PAE = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(PAE)]

from pae_boundary_realdata import DATASETS, prepare_dataset
from pae_causal_horizon import evaluate_units
from pae_shared_battery_nn import BatteryRepresentationScale, rows_from_part, concatenate_rows
from pp_extrapolation.boundary_quotient import (
    fit_boundary_quotient_pp, predict_boundary_affine, predict_boundary_quotient,
)

OUT = ROOT / "results/boundary_quotient_pp_batteries_v1"
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = [
    {"width": w, "alpha": a, "weight_decay": wd, "residual_bound": rb}
    for w, a, wd, rb in (
        (32, .1, .01, 1.), (32, 10., .01, 2.), (32, 1000., .01, 2.),
        (64, .1, .01, 2.), (64, 10., .001, 2.), (64, 10., .01, 2.),
        (64, 10., .1, 2.), (64, 1000., .01, 2.),
        (128, .1, .01, 2.), (128, 10., .001, 2.),
        (128, 10., .01, 4.), (128, 1000., .01, 2.),
        (64, 1000., .01, 4.), (64, 1000., .01, 8.),
        (128, 1000., .01, 4.), (128, 1000., .01, 8.),
        (256, 1000., .01, 4.), (256, 1000., .01, 8.),
    )
]


def full_part(split):
    return {key: np.concatenate((split["train"][key], split["val"][key])) for key in split["train"]}


def build_rows(part, scale, index):
    value = rows_from_part(part, scale, index)
    return {"x": value["x"], "margin": value["margin"], "y": value["y"],
            "units": value["units"], "dataset": value["dataset"]}


def score_by_dataset(prediction, rows, scales):
    result = {}
    for index, name in enumerate(DATASETS):
        mask = rows["dataset"] == index
        y = rows["y"][mask] * scales[name].time_scale
        p = prediction[mask] * scales[name].time_scale
        result[name] = evaluate_units(y, p, rows["units"][mask])
    return result


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); splits, audits, scales = {}, {}, {}
    train_parts, validation_parts, full_parts, source_parts = [], [], [], []
    for index, name in enumerate(DATASETS):
        split, audit = prepare_dataset(name); splits[name], audits[name] = split, audit
        scales[name] = BatteryRepresentationScale.fit(split["train"], audit["boundary"])
        train_parts.append(build_rows(split["train"], scales[name], index))
        validation_parts.append(build_rows(split["val"], scales[name], index))
        full_parts.append(build_rows(full_part(split), scales[name], index))
        source_parts.append(build_rows(split["source"], scales[name], index))
    train, validation, full, source = map(concatenate_rows, (train_parts, validation_parts, full_parts, source_parts))

    search = []
    for config in CONFIGS:
        fit = fit_boundary_quotient_pp(train, validation, seed=42, max_epochs=350, patience=55, **config)
        search.append({**config, **fit.selection})
        print("SEARCH", config, fit.selection["validation_dataset_macro_mse"], flush=True)
    chosen = min(search, key=lambda row: row["validation_dataset_macro_mse"])
    config = {key: chosen[key] for key in ("width", "alpha", "weight_decay", "residual_bound")}

    runs, predictions, affine_predictions = [], [], []
    for seed in SEEDS:
        selection_fit = fit_boundary_quotient_pp(train, validation, seed=seed, max_epochs=500, patience=70, **config)
        fit = fit_boundary_quotient_pp(
            full, full, seed=seed, max_epochs=max(selection_fit.selection["selected_epoch"], 1),
            patience=10_000, restore_best=False, **config,
        )
        prediction = predict_boundary_quotient(fit, source)
        affine = predict_boundary_affine(fit, source)
        metrics = score_by_dataset(prediction, source, scales)
        affine_metrics = score_by_dataset(affine, source, scales)
        predictions.append(prediction); affine_predictions.append(affine)
        runs.append({"seed": seed, "selection": selection_fit.selection,
                     "refit_epochs": fit.selection["selected_epoch"], "metrics": metrics,
                     "affine_only_metrics": affine_metrics})
        print("SEED", seed, {d: round(metrics[d]["pooled_r2"], 3) for d in DATASETS}, flush=True)

    ensemble = score_by_dataset(np.mean(predictions, axis=0), source, scales)
    affine_ensemble = score_by_dataset(np.mean(affine_predictions, axis=0), source, scales)
    existing = json.load(open(PAE / "results/pae_shared_battery_v1/results.json"))["summary"]["boundary_gated_shared_nn"]["datasets"]
    baseline = json.load(open(ROOT / "results/additional_real_batteries/results.json"))["datasets"]
    result = {
        "status": "retrospective relationship-shift development screen",
        "model": "exact-zero health boundary x (frozen affine quotient + bounded neural residual)",
        "selection": "18 configurations on validation only; source labels excluded",
        "selected_config": config, "search": search, "runs": runs,
        "ensemble": ensemble, "affine_only_ensemble": affine_ensemble,
        "comparators": {
            name: {
                "baseline_pp_pooled_r2": baseline[name]["summary"]["pp"]["pooled_r2"]["mean"],
                "pae_boundary_nn_pooled_r2_mean": existing[name]["pooled_r2"]["mean"],
                "pae_boundary_nn_macro_r2_mean": existing[name]["macro_unit_r2"]["mean"],
            } for name in DATASETS
        },
        "data_audits": audits, "runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", prediction=np.asarray(predictions),
                        affine=np.asarray(affine_predictions), y=source["y"],
                        units=source["units"], dataset=source["dataset"])
    print("ENSEMBLE", {d: ensemble[d]["pooled_r2"] for d in DATASETS})


if __name__ == "__main__": main()
