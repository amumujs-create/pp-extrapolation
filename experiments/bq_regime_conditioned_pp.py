#!/usr/bin/env python3
"""Validation-selected unified regime-conditioned BQ-PP on three batteries."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(ADAPTERS), str(ROOT / "experiments")]

from pae_boundary_realdata import DATASETS, prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale, concatenate_rows
from boundary_quotient_pp_batteries import build_rows, full_part, score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp, predict_boundary_quotient

OUT = ROOT / "results/bq_regime_conditioned_pp_v2"
SEEDS = (42, 43, 44, 45, 46)
SEARCH = [
    {"extra_residual_bound": extra, "regime_gate_penalty": penalty}
    for extra in (2.0, 4.0, 6.0, 8.0)
    for penalty in (0.0, 1e-4, 1e-3, 1e-2)
]
BASE = dict(width=64, alpha=1000.0, learning_rate=1e-3, weight_decay=1e-2,
            residual_bound=2.0)


def gate_summary(fit, rows):
    x = torch.as_tensor((rows["x"] - fit.center) / fit.scale, dtype=torch.float32)
    fit.model.eval()
    with torch.no_grad():
        gate = torch.sigmoid(fit.model.regime_gate(x)).squeeze(1).numpy()
    return {
        name: {
            "mean": float(np.mean(gate[rows["dataset"] == i])),
            "p90": float(np.quantile(gate[rows["dataset"] == i], .9)),
        }
        for i, name in enumerate(DATASETS)
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scales, audits = {}, {}
    parts = {key: [] for key in ("train", "validation", "full", "source")}
    for i, name in enumerate(DATASETS):
        split, audit = prepare_dataset(name)
        audits[name] = audit
        scales[name] = BatteryRepresentationScale.fit(split["train"], audit["boundary"])
        for key, part in (("train", split["train"]), ("validation", split["val"]),
                          ("full", full_part(split)), ("source", split["source"])):
            parts[key].append(build_rows(part, scales[name], i))
    rows = {key: concatenate_rows(value) for key, value in parts.items()}

    search = []
    for config in SEARCH:
        fit = fit_boundary_quotient_pp(
            rows["train"], rows["validation"], seed=42, max_epochs=500, patience=70,
            **BASE, **config,
        )
        item = {**config, **fit.selection, "validation_gate": gate_summary(fit, rows["validation"])}
        search.append(item)
        print("SEARCH", config, round(item["validation_dataset_macro_mse"], 6), flush=True)
    chosen = min(search, key=lambda item: item["validation_dataset_macro_mse"])
    config = {key: chosen[key] for key in ("extra_residual_bound", "regime_gate_penalty")}

    runs, predictions = [], []
    for seed in SEEDS:
        selection_fit = fit_boundary_quotient_pp(
            rows["train"], rows["validation"], seed=seed, max_epochs=600, patience=80,
            **BASE, **config,
        )
        epochs = max(selection_fit.selection["selected_epoch"], 1)
        fit = fit_boundary_quotient_pp(
            rows["full"], rows["full"], seed=seed, max_epochs=epochs, patience=10_000,
            restore_best=False, **BASE, **config,
        )
        prediction = predict_boundary_quotient(fit, rows["source"])
        predictions.append(prediction)
        metrics = score_by_dataset(prediction, rows["source"], scales)
        runs.append({
            "seed": seed, "selected_epoch": epochs,
            "validation_mse": selection_fit.selection["validation_dataset_macro_mse"],
            "source_gate": gate_summary(fit, rows["source"]), "metrics": metrics,
        })
        print("SEED", seed, {d: round(metrics[d]["pooled_r2"], 3) for d in DATASETS}, flush=True)

    prediction = np.mean(predictions, axis=0)
    ensemble = score_by_dataset(prediction, rows["source"], scales)
    result = {
        "status": "retrospective unified PP development; source labels excluded from selection",
        "model": "frozen affine quotient + single residual with learned regime-conditioned bound",
        "selection": "single shared configuration selected by validation dataset-macro normalized MSE",
        "base_config": BASE, "selected_config": config, "search": search, "runs": runs,
        "ensemble": ensemble,
        "dataset_mean_pooled_r2": float(np.mean([ensemble[d]["pooled_r2"] for d in DATASETS])),
        "dataset_macro_unit_r2": float(np.mean([ensemble[d]["macro_unit_r2"] for d in DATASETS])),
        "global_score_correction_bound": BASE["residual_bound"] + config["extra_residual_bound"],
        "data_audits": audits, "runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", prediction=np.asarray(predictions),
                        y=rows["source"]["y"], units=rows["source"]["units"],
                        dataset=rows["source"]["dataset"])
    print("ENSEMBLE", {d: ensemble[d]["pooled_r2"] for d in DATASETS}, flush=True)


if __name__ == "__main__":
    main()
