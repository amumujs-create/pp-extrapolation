#!/usr/bin/env python3
"""FEMTO spectrum sequence screening and five-seed endpoint evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from femto_corrected_benchmark_v2 import TRAIN, VAL, TEST
from femto_sensor_adapter_v2 import OFFICIAL
from femto_spectrum_adapter_v3 import load
from pp_extrapolation import regression_metrics
from pp_extrapolation.temporal import fit_temporal, predict_temporal
from pp_extrapolation.temporal import TemporalPriorNet
from pp_extrapolation.model import equal_group_weights

OUT = ROOT / "results/femto_spectrum_gru_v3"
SEEDS = (42, 43, 44, 45, 46)


def rows(payload, units, window: int, *, endpoint: bool, representation: str) -> dict:
    sequences, truth, groups = [], [], []
    allowed = np.isin(payload["unit"], list(units))
    for bearing in np.unique(payload["bearing"][allowed]):
        index = np.flatnonzero(payload["bearing"] == bearing)
        index = index[np.argsort(payload["recording_index"][index])]
        sensor = payload["sensor"][index]
        spectrum = payload["spectrum"][index]
        common = np.column_stack([payload["condition"][index],
            np.log1p(payload["elapsed_s"][index]) / 10.0, sensor]).astype(np.float32)
        selected = [len(index)-1] if endpoint else range(len(index))
        for end in selected:
            chosen = np.maximum(np.arange(end-window+1, end+1), 0)
            if representation == "spectrum":
                # The baseline available at an early endpoint uses only that endpoint's prefix.
                baseline = np.median(spectrum[:min(10, end+1)], axis=0)
                feature = np.column_stack([common, spectrum, spectrum-baseline]).astype(np.float32)
            elif representation == "summary":
                feature = common
            else:
                raise ValueError(representation)
            sequences.append(feature[chosen])
            truth.append(payload["y"][index[end]])
            groups.append(bearing)
    n = len(sequences)
    return {"x": np.asarray(sequences), "y": np.asarray(truth, np.float32),
            "groups": np.asarray(groups), "prior": np.zeros((n, 2), np.float32),
            "reliability": np.full((n, 2), 0.5, np.float32)}


def main() -> None:
    import torch
    torch.set_num_threads(2)
    payload = load()
    screening = []
    for representation in ("summary", "spectrum"):
        for window in (16, 32, 64):
            train = rows(payload, TRAIN, window, endpoint=False, representation=representation)
            validation = rows(payload, VAL, window, endpoint=False, representation=representation)
            fit = fit_temporal(train, validation, seed=42, mode="direct", epochs=350, patience=70)
            prediction, _ = predict_temporal(fit, validation)
            screening.append({"representation": representation, "window": window,
                              "validation_rmse": float(np.sqrt(np.mean((prediction-validation["y"])**2))),
                              "selected_epoch": fit["selection"]["epoch"]})
            print(screening[-1], flush=True)
    selected = min(screening, key=lambda row: row["validation_rmse"])
    train = rows(payload, TRAIN, selected["window"], endpoint=False, representation=selected["representation"])
    validation = rows(payload, VAL, selected["window"], endpoint=False, representation=selected["representation"])
    test = rows(payload, TEST, selected["window"], endpoint=True, representation=selected["representation"])
    assert all(abs(test["y"][i]-OFFICIAL[group]) < 1e-6 for i, group in enumerate(test["groups"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_manifest.json").write_text(json.dumps({"screening_seed": 42,
        "selection_metric": "validation RMSE", "selected": selected, "screening": screening}, indent=2)+"\n")
    predictions, runs = [], []
    for seed in SEEDS:
        fit = fit_temporal(train, validation, seed=seed, mode="direct", epochs=450, patience=90)
        prediction, _ = predict_temporal(fit, test)
        predictions.append(prediction)
        runs.append({"seed": seed, "selection": fit["selection"],
                     "metrics": regression_metrics(test["y"], prediction, test["groups"])})
        print(seed, runs[-1]["metrics"]["pooled"]["r2"], flush=True)
    predictions = np.asarray(predictions)
    result = {"status": "retrospective structural development", "selected": selected,
              "runs": runs, "ensemble": regression_metrics(test["y"], predictions.mean(0), test["groups"]),
              "seed_r2_mean": float(np.mean([r["metrics"]["pooled"]["r2"] for r in runs])),
              "seed_r2_sample_sd": float(np.std([r["metrics"]["pooled"]["r2"] for r in runs], ddof=1))}
    (OUT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "predictions.npz", y=test["y"], groups=test["groups"], predictions=predictions)
    # Standard final refit: selection is frozen above, then all six Learning bearings
    # are used for exactly the selected number of epochs without test-based stopping.
    development = rows(payload, TRAIN | VAL, selected["window"], endpoint=False,
                       representation=selected["representation"])
    refit_predictions, refit_runs = [], []
    fixed_epochs = max(1, int(selected["selected_epoch"]))
    for seed in SEEDS:
        torch.manual_seed(seed)
        center = development["x"][:, -1].mean(0)
        scale = development["x"][:, -1].std(0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        cap = max(float(np.max(development["y"])), 1.0)
        tx = torch.tensor((development["x"]-center)/scale)
        target = torch.tensor(development["y"]/cap)
        weight = torch.tensor(equal_group_weights(development["groups"]), dtype=torch.float32)
        model = TemporalPriorNet(tx.shape[-1], mode="direct")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
        rng = np.random.default_rng(seed)
        dummy_prior = torch.zeros((len(tx), 2))
        dummy_reliability = torch.full((len(tx), 2), .5)
        for _ in range(fixed_epochs):
            model.train()
            order = rng.permutation(len(tx))
            for start in range(0, len(tx), 512):
                index = order[start:start+512]
                estimate = model(tx[index], dummy_prior[index], dummy_reliability[index])[0]
                loss = torch.mean(weight[index]*(estimate-target[index]).square())
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
        fit = {"model": model, "center": center, "scale": scale, "cap": cap,
               "selection": {"seed": seed, "mode": "direct", "epoch": fixed_epochs,
                             "validation_mse": None}}
        prediction, _ = predict_temporal(fit, test)
        refit_predictions.append(prediction)
        refit_runs.append({"seed": seed, "metrics": regression_metrics(test["y"], prediction, test["groups"])})
    refit_predictions = np.asarray(refit_predictions)
    result["all_learning_refit"] = {
        "fixed_epochs": fixed_epochs, "runs": refit_runs,
        "ensemble": regression_metrics(test["y"], refit_predictions.mean(0), test["groups"]),
        "seed_r2_mean": float(np.mean([r["metrics"]["pooled"]["r2"] for r in refit_runs])),
        "seed_r2_sample_sd": float(np.std([r["metrics"]["pooled"]["r2"] for r in refit_runs], ddof=1)),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "all_learning_refit_predictions.npz", y=test["y"],
                        groups=test["groups"], predictions=refit_predictions)
    print("all_learning_refit", json.dumps(result["all_learning_refit"]["ensemble"]["pooled"]), flush=True)
    print(json.dumps({"selected": selected, "ensemble": result["ensemble"]["pooled"],
                      "seed_mean": result["seed_r2_mean"], "seed_sd": result["seed_r2_sample_sd"]}, indent=2))


if __name__ == "__main__":
    main()
