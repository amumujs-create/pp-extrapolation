#!/usr/bin/env python3
"""FEMTO PP with a causal GRU correction around an affine prior."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from femto_corrected_gru_v2 import part
from femto_corrected_benchmark_v2 import TRAIN, VAL, TEST
from femto_sensor_adapter_v2 import load
from pp_extrapolation import regression_metrics
from pp_extrapolation.temporal import fit_temporal, predict_temporal

OUT = ROOT / "results/femto_affine_gru_pp_v3"
SEEDS = (42, 43, 44, 45, 46)


def attach_affine_prior(train, validation, test):
    current_train = train["x"][:, -1]
    current_validation = validation["x"][:, -1]
    current_test = test["x"][:, -1]
    center = current_train.mean(0)
    scale = current_train.std(0)
    scale[scale < 1e-8] = 1.0
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        model = Ridge(alpha=alpha).fit((current_train-center)/scale, train["y"])
        prediction = np.maximum(model.predict((current_validation-center)/scale), 0.0)
        candidates.append((float(np.mean((prediction-validation["y"])**2)), alpha, model))
    _, alpha, model = min(candidates, key=lambda row: row[0])
    for split, current in ((train, current_train), (validation, current_validation), (test, current_test)):
        prediction = np.maximum(model.predict((current-center)/scale), 0.0).astype(np.float32)
        split["prior"] = np.column_stack([prediction, prediction]).astype(np.float32)
        split["reliability"] = np.full((len(prediction), 2), 0.5, np.float32)
    return alpha


def main():
    import torch
    torch.set_num_threads(2)
    payload = load()
    train = part(payload, TRAIN)
    validation = part(payload, VAL)
    test = part(payload, TEST, True)
    alpha = attach_affine_prior(train, validation, test)
    predictions, runs = [], []
    for seed in SEEDS:
        fit = fit_temporal(train, validation, seed=seed, mode="corrected", epochs=450, patience=90)
        prediction, gate = predict_temporal(fit, test)
        predictions.append(prediction)
        runs.append({"seed": seed, "selection": fit["selection"],
                     "metrics": regression_metrics(test["y"], prediction, test["groups"]),
                     "mean_gate": gate.mean(0).tolist()})
        print(seed, runs[-1]["metrics"]["pooled"]["r2"], flush=True)
    predictions = np.asarray(predictions)
    result = {"status": "retrospective structural development", "affine_alpha": alpha,
              "model": "frozen affine current-state prior + causal GRU bounded correction",
              "runs": runs,
              "ensemble": regression_metrics(test["y"], predictions.mean(0), test["groups"]),
              "seed_r2_mean": float(np.mean([r["metrics"]["pooled"]["r2"] for r in runs])),
              "seed_r2_sample_sd": float(np.std([r["metrics"]["pooled"]["r2"] for r in runs], ddof=1))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=test["y"], groups=test["groups"], predictions=predictions)
    print(json.dumps({"ensemble": result["ensemble"]["pooled"],
                      "seed_mean": result["seed_r2_mean"], "seed_sd": result["seed_r2_sample_sd"]}, indent=2))


if __name__ == "__main__":
    main()
