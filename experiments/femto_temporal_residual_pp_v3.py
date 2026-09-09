#!/usr/bin/env python3
"""Corrected FEMTO comparison: direct TCN versus temporal-residual PP."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from femto_corrected_benchmark_v2 import TRAIN, VAL, TEST
from femto_sensor_adapter_v2 import OFFICIAL, load
from pp_extrapolation import regression_metrics
from pp_extrapolation.temporal_residual_pp import fit_temporal_residual_pp, predict_temporal_residual_pp

OUT = ROOT / "results/femto_temporal_residual_pp_v3"
SELECTION_SEEDS = (42,)
FINAL_SEEDS = (42, 43, 44, 45, 46)
WINDOW = 64
CONFIGS = tuple(
    {"mode": mode, "width": width, "learning_rate": lr,
     "weight_decay": wd, "residual_bound": bound,
     "support_decay": decay, "dropout": dropout}
    for mode in ("direct", "pp")
    for width in (16,)
    for lr in (3e-4, 1e-3)
    for wd in (0.05,)
    for bound in ((0.25, 0.5) if mode == "pp" else (0.5,))
    for decay in ((0.0, 0.1) if mode == "pp" else (0.0,))
    for dropout in (0.0,)
)


def build_rows(payload, units, *, endpoint: bool) -> dict:
    sequences, masks, truth, groups = [], [], [], []
    allowed = np.isin(payload["unit"], list(units))
    for bearing in np.unique(payload["bearing"][allowed]):
        index = np.flatnonzero(payload["bearing"] == bearing)
        index = index[np.argsort(payload["recording_index"][index])]
        elapsed = payload["elapsed_s"][index]
        values = np.column_stack([
            payload["condition"][index],
            np.log1p(elapsed) / 10.0,
            payload["sensor"][index],
        ]).astype(np.float32)
        chosen = [len(index) - 1] if endpoint else range(len(index))
        for end in chosen:
            start = max(0, end - WINDOW + 1)
            observed = values[start:end + 1]
            sequence = np.zeros((WINDOW, values.shape[1]), np.float32)
            mask = np.zeros(WINDOW, bool)
            sequence[-len(observed):] = observed
            mask[-len(observed):] = True
            sequences.append(sequence)
            masks.append(mask)
            truth.append(payload["y"][index[end]])
            groups.append(bearing)
    return {"x": np.asarray(sequences), "mask": np.asarray(masks),
            "y": np.asarray(truth, np.float32), "groups": np.asarray(groups)}


def rmse(truth, prediction) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def main() -> None:
    import torch
    torch.set_num_threads(2)
    payload = load()
    train = build_rows(payload, TRAIN, endpoint=False)
    validation = build_rows(payload, VAL, endpoint=False)
    test = build_rows(payload, TEST, endpoint=True)
    assert all(abs(test["y"][i] - OFFICIAL[group]) < 1e-6 for i, group in enumerate(test["groups"]))

    screening = []
    for number, config in enumerate(CONFIGS, 1):
        predictions = []
        epochs = []
        for seed in SELECTION_SEEDS:
            fit = fit_temporal_residual_pp(train, validation, seed=seed,
                                           max_epochs=100, patience=20, **config)
            predictions.append(predict_temporal_residual_pp(fit, validation))
            epochs.append(fit.selection["selected_epoch"])
        seed_rmse = [rmse(validation["y"], value) for value in predictions]
        screening.append({**config, "mean_seed_validation_rmse": float(np.mean(seed_rmse)),
                          "ensemble_validation_rmse": rmse(validation["y"], np.mean(predictions, axis=0)),
                          "seed_validation_rmse": seed_rmse, "selected_epochs": epochs})
        print(number, len(CONFIGS), config["mode"], screening[-1]["mean_seed_validation_rmse"], flush=True)

    selected = {}
    for mode in ("direct", "pp"):
        selected[mode] = min((row for row in screening if row["mode"] == mode),
                             key=lambda row: (row["mean_seed_validation_rmse"], row["width"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_manifest.json").write_text(json.dumps({
        "status": "retrospective development on observed FEMTO test",
        "input": "corrected acceleration columns 4/5, causal 64-recording history",
        "split": {"train": sorted(TRAIN), "validation": sorted(VAL), "test": sorted(TEST)},
        "selection_objective": "validation RMSE at screening seed 42; final robustness uses seeds 42-46",
        "selected": selected, "all_search": screening,
    }, indent=2) + "\n")

    results = {}
    for mode, choice in selected.items():
        config = {key: choice[key] for key in ("mode", "width", "learning_rate", "weight_decay",
                                                "residual_bound", "support_decay", "dropout")}
        predictions, runs = [], []
        for seed in FINAL_SEEDS:
            fit = fit_temporal_residual_pp(train, validation, seed=seed,
                                           max_epochs=220, patience=45, **config)
            prediction = predict_temporal_residual_pp(fit, test)
            predictions.append(prediction)
            runs.append({"seed": seed, "selection": fit.selection,
                         "metrics": regression_metrics(test["y"], prediction, test["groups"])})
        predictions = np.asarray(predictions)
        results[mode] = {"selected_config": config, "runs": runs,
                         "ensemble": regression_metrics(test["y"], predictions.mean(0), test["groups"]),
                         "seed_r2_mean": float(np.mean([r["metrics"]["pooled"]["r2"] for r in runs])),
                         "seed_r2_sample_sd": float(np.std([r["metrics"]["pooled"]["r2"] for r in runs], ddof=1))}
        np.savez_compressed(OUT / f"{mode}_predictions.npz", y=test["y"], groups=test["groups"], predictions=predictions)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({mode: {"ensemble": row["ensemble"]["pooled"],
                                  "seed_mean": row["seed_r2_mean"], "seed_sd": row["seed_r2_sample_sd"]}
                      for mode, row in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
