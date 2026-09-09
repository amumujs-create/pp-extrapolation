#!/usr/bin/env python3
"""Known-boundary PP route for the observed NASA milling material shift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]

from milling_locked_transfer import FEATURES, subset
from nasa_milling_causal import BOUNDARY, prepare_causal_milling
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import certify_categorical_regime, regression_metrics

OUT = ROOT / "results" / "milling_boundary_quotient_route_v1"
SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
CONFIGS = tuple(
    {"width": w, "learning_rate": lr, "weight_decay": wd}
    for w in (32, 64) for lr in (5e-4, 1e-3) for wd in (.1, 2.)
)
BOUNDARY_OFFSETS = tuple(float(x) for x in np.arange(-.05, .101, .01))


def quotient(rows, effective_boundary):
    health = rows["x"][:, FEATURES.index("health")]
    rate = np.maximum(rows["x"][:, FEATURES.index("rate")], 1e-6)
    return np.maximum(float(effective_boundary) - health, 0.0) / rate


def main():
    torch.set_num_threads(2)
    raw, audit = prepare_causal_milling()
    cut = float(np.quantile(raw["train"]["health"], .60))
    train = subset(raw["train"], raw["train"]["health"] <= cut)
    validation = subset(raw["validation"], raw["validation"]["health"] > cut)
    test = subset(raw["source"], raw["source"]["health"] > cut)
    # Sparse inspections cross the official boundary between two observations.
    # Select a small effective-margin correction by validation MAE, which is
    # less dominated by one point than RMSE when validation contains four rows.
    boundary_search = []
    for offset_candidate in BOUNDARY_OFFSETS:
        candidate = float(BOUNDARY) + offset_candidate
        prediction = quotient(validation, candidate)
        boundary_search.append({
            "offset": offset_candidate,
            "effective_boundary": candidate,
            "validation_mae": float(np.mean(np.abs(prediction-validation["y"]))),
            "validation_rmse": float(np.sqrt(np.mean((prediction-validation["y"])**2))),
        })
    boundary_choice = min(boundary_search, key=lambda row: (row["validation_mae"], abs(row["offset"])))
    effective_boundary = boundary_choice["effective_boundary"]
    q_train = quotient(train, effective_boundary)
    q_validation = quotient(validation, effective_boundary)

    # The NN learns only the signed discrepancy around the physical quotient.
    # A positive offset lets the existing direct-RUL trainer retain its exact
    # optimization protocol; it is removed again after prediction.
    offset = float(max(1.0, -np.min(train["y"] - q_train) + 1.0))
    residual_train = {**train, "y": (train["y"] - q_train + offset).astype(np.float32)}
    residual_validation = {**validation, "y": (validation["y"] - q_validation + offset).astype(np.float32)}
    search = []
    for config in CONFIGS:
        predictions = []
        for seed in SELECTION_SEEDS:
            fit = fit_plain(residual_train, residual_validation, seed=seed, **config)
            predictions.append(q_validation + predict_plain(fit, validation["x"]) - offset)
        prediction = np.mean(predictions, axis=0)
        search.append({**config, "validation_rmse": float(np.sqrt(np.mean((prediction-validation["y"])**2)))})
    selected = min(search, key=lambda row: (row["validation_rmse"], row["width"], row["weight_decay"]))

    regime = certify_categorical_regime(np.ones(len(train["y"]), dtype=int), np.full(len(test["y"]), 2))
    if regime.accepted:
        raise RuntimeError("material-2 must remain unseen in material-1 training")
    # The unseen-material gate is frozen before evaluating target labels.  It
    # disables the learned discrepancy but preserves the known-boundary prior.
    q_test = quotient(test, effective_boundary)
    test_predictions = np.repeat(q_test[None, :], len(FINAL_SEEDS), axis=0)
    payload = {
        "status": "post-test model development on the already-observed milling split",
        "protocol": "joint train/validation residual tuning; label-free unseen-material gate; test after selection",
        "known_failure_boundary": float(BOUNDARY),
        "boundary_offset_search": boundary_search,
        "selected_inspection_offset": boundary_choice["offset"],
        "effective_prediction_boundary": effective_boundary,
        "formula": "RUL=(official_boundary+validation_inspection_offset-health)/causal_rate + seen_material*NN_residual",
        "selection_seeds": SELECTION_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "residual_search": search,
        "selected_residual_hyperparameters": {k: selected[k] for k in ("width", "learning_rate", "weight_decay")},
        "material_certificate": regime.__dict__,
        "nn_residual_enabled_on_test": False,
        "validation_quotient": regression_metrics(validation["y"], q_validation, validation["groups"]),
        "test": regression_metrics(test["y"], test_predictions.mean(0), test["groups"]),
        "source_audit": audit,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=test["y"], groups=test["groups"], prediction=test_predictions)
    print(json.dumps({"selected": payload["selected_residual_hyperparameters"],
                      "validation": payload["validation_quotient"]["pooled"],
                      "test": payload["test"]["pooled"]}, indent=2))


if __name__ == "__main__":
    main()
