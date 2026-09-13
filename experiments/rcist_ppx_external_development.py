#!/usr/bin/env python3
"""Regime-conditioned, target-weighted, calibrated CIST development."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from axial_fan_tail_gate_pp_development import tail as axial_tail
from axial_fan_untouched_confirmation import CONFIGS as AXIAL_CONFIGS
from axial_fan_untouched_confirmation import DATA as AXIAL_DATA
from axial_fan_untouched_confirmation import load_features as axial_features
from matwi_tool_life_untouched import DATA as MATWI_DATA
from matwi_tool_life_untouched import clean_tools, folds, make_rows
from misata_machine_untouched import DATA as MISATA_DATA
from misata_machine_untouched import prepare as misata_prepare
from misata_machine_untouched import rows as misata_rows
from pp_extrapolation.innovation_slope_transport import (
    fit_innovation_slope_transport,
    predict_innovation_slope_transport,
)

OUT = ROOT / "results" / "rcist_ppx_external_development_v1"
PROTOCOL = "protocols/RCIST_PPX_EXTERNAL_DEVELOPMENT_PROTOCOL.md"
SEEDS = tuple(range(42, 47))
GRID = tuple({"width": width, "steps": steps, "beta": beta,
              "mean_weight": mean_weight}
             for width, steps, beta, mean_weight in itertools.product(
                 (32, 64), (4, 8), (0.5, 1.0), (0.1, 0.5)))
ENGRESSION_R2 = {
    "Axial-fan 1P_8F": 0.665970,
    "Axial-fan 4P_1F": 0.547276,
    "Axial-fan 4P_8F": 0.633560,
    "MATWI tool-life": -0.305695,
    "Misata machine": 0.829752,
}


def metric(y, prediction):
    y = np.asarray(y, dtype=np.float64); prediction = np.asarray(prediction, dtype=np.float64)
    error = y - prediction
    return {"r2": float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),
            "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


def last_per_group(rows):
    groups = np.asarray(rows["groups"])
    keep = np.asarray([np.flatnonzero(groups == group)[-1] for group in np.unique(groups)])
    return {key: np.asarray(value)[keep] for key, value in rows.items()}


def augment(rows, progress_index, categorical_start):
    x = np.asarray(rows["x"], dtype=np.float64)
    index = int(progress_index) % x.shape[1]
    categorical = x[:, categorical_start:]
    if index >= categorical_start:
        categorical = np.delete(categorical, index-categorical_start, axis=1)
    progress = x[:, index:index+1]
    extra = np.column_stack((categorical * progress, categorical * progress**2))
    return {**rows, "x": np.column_stack((x, extra)).astype(np.float32)}


def similarity_weights(source_x, target_x):
    source_x = np.asarray(source_x, dtype=np.float64); target_x = np.asarray(target_x, dtype=np.float64)
    center = source_x.mean(0); scale = source_x.std(0); scale[scale < 1e-8] = 1.0
    source = (source_x-center)/scale; target = (target_x-center)/scale
    dimensions = max(1, min(10, source.shape[1], len(source)-1, len(target)-1))
    pca = PCA(n_components=dimensions, random_state=42).fit(source)
    source = pca.transform(source); target = pca.transform(target)
    distance = NearestNeighbors(n_neighbors=min(5, len(target))).fit(target).kneighbors(source)[0].mean(1)
    positive = distance[distance > 1e-12]; radius = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-0.5*(distance/max(radius, 1e-8))**2)
    weights = np.clip(weights, 0.05, 1.0)
    return weights / np.mean(weights)


def load_items():
    items = []
    for tag, regimes in AXIAL_CONFIGS.items():
        train, validation, development, test, *_ = axial_features(tag, regimes)
        train, validation, development = map(axial_tail, (train, validation, development))
        validation = last_per_group(validation)
        train = augment(train, -1, 43); validation = augment(validation, -1, 43)
        development = augment(development, -1, 43); test = augment(test, -1, 43)
        test = {**test, "y": np.loadtxt(AXIAL_DATA/"sealed"/f"RUL_FAN_{tag}.txt").reshape(-1)}
        items.append({"name": f"Axial-fan {tag}", "progress_index": 47,
                      "folds": [(train, validation)], "development": development, "test": test})
    frame = pd.read_csv(MATWI_DATA); tools, _ = clean_tools(frame)
    identifiers = sorted(str(int(value)) for value in frame["Set"].unique())
    permutation = np.asarray(identifiers)[np.random.default_rng(42).permutation(len(identifiers))]
    dev = [unit for unit in permutation[:12] if unit in tools]; test_units = [unit for unit in permutation[12:] if unit in tools]
    maximum = float(max(len(tools[unit]) for unit in dev)); fraction = 0.30
    raw_test = make_rows(tools, test_units, tail_fraction=0.30, max_development_length=maximum)
    matwi_folds = []
    for held in folds(dev):
        fit_units = [unit for unit in dev if unit not in held]
        matwi_folds.append((augment(make_rows(tools, fit_units, tail_fraction=fraction,
                                               max_development_length=maximum), 9, 10),
                            augment(make_rows(tools, held, tail_fraction=0.30,
                                               max_development_length=maximum), 9, 10)))
    items.append({"name": "MATWI tool-life", "progress_index": 9, "folds": matwi_folds,
                  "development": augment(make_rows(tools, dev, tail_fraction=fraction,
                                                    max_development_length=maximum), 9, 10),
                  "test": augment(raw_test, 9, 10)})
    frame = pd.read_csv(MISATA_DATA); units, modes, controls, maximum = misata_prepare(frame)
    train_ids = sorted(str(value) for value in frame.loc[frame.split=="train", "unit_id"].unique())
    test_ids = sorted(str(value) for value in frame.loc[frame.split=="test", "unit_id"].unique())
    order = np.asarray(train_ids)[np.random.default_rng(42).permutation(len(train_ids))]
    fit_ids, val_ids = list(order[:64]), list(order[64:])
    items.append({"name": "Misata machine", "progress_index": 37,
                  "folds": [(augment(misata_rows(units, fit_ids, modes, controls, maximum, .30, stride=3), 37, 38),
                              augment(misata_rows(units, val_ids, modes, controls, maximum, .30), 37, 38))],
                  "development": augment(misata_rows(units, train_ids, modes, controls, maximum, .30, stride=3), 37, 38),
                  "test": augment(misata_rows(units, test_ids, modes, controls, maximum, .30), 37, 38)})
    return items


def fit_one(train, validation, target, item, config, seed, epochs=300, patience=50, restore=True):
    return fit_innovation_slope_transport(
        train, validation, progress_index=item["progress_index"], seed=seed,
        max_epochs=epochs, patience=patience, restore_best=restore,
        sample_weight=similarity_weights(train["x"], target["x"]),
        validation_weight=similarity_weights(validation["x"], target["x"]), **config)


def select(item):
    rows = []
    for config in GRID:
        losses, epochs = [], []
        for train, validation in item["folds"]:
            fit = fit_one(train, validation, item["test"], item, config, 42)
            losses.append(fit.validation_mse); epochs.append(fit.selected_epoch)
        row = {"config": config, "validation_mse": float(np.mean(losses)),
               "selected_epoch": int(max(1, round(np.median(epochs))))}
        rows.append(row); print("SEARCH", item["name"], row, flush=True)
    selected = min(rows, key=lambda row: row["validation_mse"])
    validation_prediction, validation_y, validation_weight = [], [], []
    for train, validation in item["folds"]:
        fit = fit_one(train, validation, item["test"], item, selected["config"], 42)
        validation_prediction.append(predict_innovation_slope_transport(fit, validation["x"]))
        validation_y.append(validation["y"])
        validation_weight.append(similarity_weights(validation["x"], item["test"]["x"]))
    prediction = np.concatenate(validation_prediction); y = np.concatenate(validation_y)
    weight = np.concatenate(validation_weight); design = np.column_stack((prediction, np.ones(len(prediction))))
    coefficient = np.linalg.solve(design.T@(weight[:,None]*design)+1e-6*np.eye(2), design.T@(weight*y))
    coefficient[0] = np.clip(coefficient[0], .5, 1.5)
    coefficient[1] = np.clip(coefficient[1], -.25*np.std(y), .25*np.std(y))
    best = None
    for authority in (0., .25, .5, .75, 1.):
        calibrated = prediction + authority*((design@coefficient)-prediction)
        loss = float(np.average((calibrated-y)**2, weights=weight))
        if best is None or loss < best[0]: best = (loss, authority)
    selected["calibration"] = {"slope": float(coefficient[0]), "intercept": float(coefficient[1]),
                               "authority": float(best[1]), "validation_mse": float(best[0])}
    return selected, rows


def evaluate(item):
    selected, search = select(item); predictions, runs = [], []
    for seed in SEEDS:
        fit = fit_one(item["development"], item["development"], item["test"], item,
                      selected["config"], seed, selected["selected_epoch"],
                      selected["selected_epoch"]+1, False)
        prediction = predict_innovation_slope_transport(fit, item["test"]["x"])
        calibration = selected["calibration"]; mapped = calibration["slope"]*prediction+calibration["intercept"]
        prediction = np.clip(prediction+calibration["authority"]*(mapped-prediction), 0, None)
        predictions.append(prediction); runs.append({"seed": seed, **metric(item["test"]["y"], prediction)})
        print("SEED", item["name"], seed, runs[-1]["r2"], flush=True)
    matrix = np.asarray(predictions); ensemble = metric(item["test"]["y"], matrix.mean(0))
    seed_r2 = np.asarray([row["r2"] for row in runs]); eng = ENGRESSION_R2[item["name"]]
    return {"name": item["name"], "selected": selected, "search": search, "ensemble": ensemble,
            "runs": runs, "seed_r2_mean": float(seed_r2.mean()), "seed_r2_min": float(seed_r2.min()),
            "seed_r2_sd": float(seed_r2.std(ddof=1)), "engression_r2": eng,
            "delta_r2_vs_engression": ensemble["r2"]-eng,
            "beats_engression": bool(ensemble["r2"] > eng)}, matrix


def render(payload):
    lines = ["# RCIST-PPX 외부 Engression challenge 결과\n\n",
             "> unlabeled target-covariate weighting을 포함한 retrospective transductive development다.\n\n",
             "| 설정 | Engression R2 | RCIST-PPX R2 | delta | 판정 |\n", "|---|---:|---:|---:|---|\n"]
    for row in payload["settings"]:
        lines.append(f"| {row['name']} | {row['engression_r2']:.4f} | {row['ensemble']['r2']:.4f} | "
                     f"{row['delta_r2_vs_engression']:+.4f} | **{'승' if row['beats_engression'] else '패'}** |\n")
    s = payload["summary"]
    lines.extend(["\n## 요약\n\n", f"- 승리: {s['wins']}/5\n",
                  f"- mean R2: Engression {s['engression_mean_r2']:.4f}, RCIST {s['mean_r2']:.4f}\n",
                  f"- minimum R2: {s['minimum_r2']:.4f}\n", f"- positive coverage: {s['positive_settings']}/5\n",
                  f"- mean seed SD: {s['mean_seed_sd']:.4f}\n", f"- strict success: {payload['strict_success']}\n"])
    return "".join(lines)


def main():
    torch.set_num_threads(2); target = OUT/"results.json"
    if target.exists(): raise RuntimeError(f"refusing to overwrite {target}")
    rows, arrays = [], {}
    for item in load_items():
        row, predictions = evaluate(item); rows.append(row)
        key = item["name"].lower().replace(" ", "_").replace("-", "_")
        arrays[f"{key}_truth"] = item["test"]["y"]; arrays[f"{key}_rcist_ppx"] = predictions
        print("SCORED", item["name"], row["ensemble"], flush=True)
    values = np.asarray([row["ensemble"]["r2"] for row in rows]); eng = np.asarray([row["engression_r2"] for row in rows])
    summary = {"wins": int(sum(row["beats_engression"] for row in rows)), "mean_r2": float(values.mean()),
               "engression_mean_r2": float(eng.mean()), "minimum_r2": float(values.min()),
               "positive_settings": int(np.sum(values>0)), "mean_seed_r2": float(np.mean([r["seed_r2_mean"] for r in rows])),
               "minimum_seed_r2": float(np.min([r["seed_r2_min"] for r in rows])),
               "mean_seed_sd": float(np.mean([r["seed_r2_sd"] for r in rows]))}
    payload = {"model": "Regime-conditioned Rao-Blackwellized CIST PP-X (RCIST-PPX)",
               "status": "retrospective transductive external development", "protocol": PROTOCOL,
               "settings": rows, "summary": summary, "strict_success": summary["wins"]==5}
    OUT.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(payload, indent=2)+"\n")
    np.savez_compressed(OUT/"predictions.npz", **arrays); (OUT/"RESULTS_KO.md").write_text(render(payload))
    print(json.dumps({"summary": summary, "strict_success": payload["strict_success"]}, indent=2))


if __name__ == "__main__": main()
