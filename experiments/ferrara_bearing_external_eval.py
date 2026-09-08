#!/usr/bin/env python3
"""Frozen one-shot Ferrara bearing evaluation; do not tune on E5/E6."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / ".benchmark_deps")]
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import (
    fit_boundary_quotient_pp, predict_boundary_affine,
    predict_boundary_quotient, regression_metrics,
)


SEEDS = (42, 43, 44, 45, 46)
LOADS = {"E1": 4.0, "E2": 4.0, "E3": 4.0, "E4": 3.0, "E5": 4.7, "E6": 5.0}
TRAIN, VALIDATION, TEST = ("E1", "E2", "E3"), ("E4",), ("E5", "E6")
FEATURE_DIR = ROOT / "data" / "ferrara_bearing" / "features"
OUT = ROOT / "results" / "ferrara_bearing_external_locked_v1"


def causal_unit(unit: str) -> dict:
    z = np.load(FEATURE_DIR / f"{unit}.npz", allow_pickle=False)
    wave = np.asarray(z["values"], dtype=np.float64)
    raw_peak, rms = wave[:, 0], wave[:, 1]
    hits = np.flatnonzero(raw_peak >= 20.0)
    event = int(hits[0]) if len(hits) else len(wave) - 1
    wave = wave[:event + 1]
    raw_peak, rms = wave[:, 0], wave[:, 1]
    running_peak = np.maximum.accumulate(raw_peak)
    n = len(wave)
    def slope(lag: int):
        out = np.zeros(n)
        out[lag:] = (running_peak[lag:] - running_peak[:-lag]) / (5.0 * lag)
        return out
    count = np.arange(1, n + 1, dtype=np.float64)
    mean = np.cumsum(rms) / count
    second = np.cumsum(rms * rms) / count
    expanding_std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    elapsed = 5.0 * np.arange(n, dtype=np.float64)
    x = np.column_stack([
        raw_peak / 20.0, running_peak / 20.0, np.maximum(20.0 - running_peak, 0.0) / 20.0,
        wave[:, 1:], slope(1), slope(3), slope(6), mean, expanding_std,
        np.log1p(count), elapsed, np.full(n, LOADS[unit]), np.full(n, 40.0),
    ])
    return {"unit": unit, "x": x, "margin": np.maximum(20.0 - running_peak, 0.0),
            "y": 5.0 * (event - np.arange(n)), "coordinate": running_peak,
            "event": event, "n_original": int(len(z["values"])),
            "endpoint_verified": bool(len(hits))}


def rows(cells: dict[str, dict], units: tuple[str, ...], portion: str) -> dict:
    parts = []
    for unit in units:
        c = cells[unit]; split = max(1, int(np.floor(0.70 * len(c["y"]))))
        idx = np.arange(split) if portion == "prefix" else np.arange(split, len(c["y"]))
        parts.append({"x": c["x"][idx], "margin": c["margin"][idx], "y": c["y"][idx],
                      "groups": np.asarray([unit] * len(idx)), "units": np.asarray([unit] * len(idx)),
                      "dataset": np.asarray(["ferrara"] * len(idx)), "coordinate": c["coordinate"][idx]})
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def fit_ridge(train: dict, validation: dict):
    candidates = []
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        scaler = StandardScaler().fit(train["x"]); model = Ridge(alpha=alpha).fit(scaler.transform(train["x"]), train["y"])
        pred = np.maximum(model.predict(scaler.transform(validation["x"])), 0.0)
        candidates.append((float(np.mean((pred - validation["y"]) ** 2)), alpha))
    return min(candidates), candidates


def main() -> None:
    torch.set_num_threads(2)
    cells = {unit: causal_unit(unit) for unit in LOADS}
    audit = {u: {k: v for k, v in c.items() if k in ("event", "n_original", "endpoint_verified")} for u, c in cells.items()}
    gates = {
        "all_test_endpoints_verified": all(cells[u]["endpoint_verified"] for u in TEST),
        "test_tail_at_least_30": all(len(cells[u]["y"]) - int(np.floor(.7 * len(cells[u]["y"]))) >= 30 for u in TEST),
    }
    if not all(gates.values()):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "results.json").write_text(json.dumps({"status": "inconclusive", "audit": audit, "gates": gates}, indent=2) + "\n")
        raise RuntimeError(f"predeclared data gate failed: {gates}")
    tr = rows(cells, TRAIN, "prefix"); va = rows(cells, VALIDATION, "tail")
    full = rows(cells, TRAIN + VALIDATION, "prefix"); te = rows(cells, TEST, "tail")
    pp_predictions, affine_predictions, mlp_predictions, runs = [], [], [], []
    for seed in SEEDS:
        selected_pp = fit_boundary_quotient_pp(tr, va, seed=seed, residual_bound=.5,
                                                max_epochs=300, patience=50)
        pp_epoch = max(int(selected_pp.selection["selected_epoch"]), 0)
        final_pp = fit_boundary_quotient_pp(full, full, seed=seed, residual_bound=.5,
                                             max_epochs=pp_epoch, patience=10000, restore_best=False)
        pp = predict_boundary_quotient(final_pp, te); affine = predict_boundary_affine(final_pp, te)
        selected_mlp = fit_plain(tr, va, seed=seed, max_epochs=300, patience=50)
        mlp_epoch = max(int(selected_mlp["selected_epoch"]), 1)
        final_mlp = fit_plain(full, full, seed=seed, max_epochs=mlp_epoch,
                              patience=10000, restore_best=False)
        mlp = predict_plain(final_mlp, te["x"], clip_to_train_max=False)
        pp_predictions.append(pp); affine_predictions.append(affine); mlp_predictions.append(mlp)
        runs.append({"seed": seed, "pp_epoch": pp_epoch, "mlp_epoch": mlp_epoch,
                     "pp": regression_metrics(te["y"], pp, te["groups"]),
                     "affine": regression_metrics(te["y"], affine, te["groups"]),
                     "mlp": regression_metrics(te["y"], mlp, te["groups"])})
        print("seed", seed, "pp", runs[-1]["pp"]["pooled"]["r2"], "mlp", runs[-1]["mlp"]["pooled"]["r2"], flush=True)
    pp = np.asarray(pp_predictions); affine = np.asarray(affine_predictions); mlp = np.asarray(mlp_predictions)
    (_, ridge_alpha), ridge_search = fit_ridge(tr, va)
    ridge_scaler = StandardScaler().fit(full["x"]); ridge = Ridge(alpha=ridge_alpha).fit(ridge_scaler.transform(full["x"]), full["y"])
    ridge_prediction = np.maximum(ridge.predict(ridge_scaler.transform(te["x"])), 0.0)
    train_lo, train_hi = float(full["coordinate"].min()), float(full["coordinate"].max())
    distance = np.maximum(np.maximum(train_lo - te["coordinate"], te["coordinate"] - train_hi), 0.0)
    result = {
        "status": "one-shot frozen external-trajectory evaluation",
        "split": {"train": TRAIN, "validation": VALIDATION, "test": TEST},
        "audit": audit, "gates": gates, "seeds": SEEDS, "runs": runs,
        "ensemble": {
            "pp": regression_metrics(te["y"], pp.mean(0), te["groups"]),
            "boundary_affine": regression_metrics(te["y"], affine.mean(0), te["groups"]),
            "plain_mlp": regression_metrics(te["y"], mlp.mean(0), te["groups"]),
            "ridge": regression_metrics(te["y"], ridge_prediction, te["groups"]),
        },
        "ridge": {"selected_alpha": ridge_alpha, "validation_search_mse_alpha": ridge_search},
        "state_extrapolation": {"train_coordinate_interval_g": [train_lo, train_hi],
                                "test_outside_fraction": float(np.mean(distance > 0)),
                                "median_outside_distance_g": float(np.median(distance)),
                                "qualifies_strict_scalar_state_extrapolation": bool(np.mean(distance > 0) >= .5)},
    }
    primary = result["ensemble"]
    result["primary_success"] = bool(primary["pp"]["pooled"]["r2"] > 0 and
                                     primary["pp"]["pooled"]["rmse"] < primary["plain_mlp"]["pooled"]["rmse"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=te["y"], groups=te["groups"], pp=pp,
                        affine=affine, mlp=mlp, ridge=ridge_prediction, coordinate=te["coordinate"], distance=distance)
    print(json.dumps({"ensemble": result["ensemble"], "state_extrapolation": result["state_extrapolation"],
                      "primary_success": result["primary_success"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

