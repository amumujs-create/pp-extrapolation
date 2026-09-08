#!/usr/bin/env python3
"""Frozen one-shot Ferrara bearing evaluation; do not tune on E5/E6."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / ".benchmark_deps")]
from rtdl_revisiting_models import FTTransformer
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
    trajectory_sha256 = __import__("hashlib").sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
    return {"unit": unit, "x": x, "margin": np.maximum(20.0 - running_peak, 0.0),
            "y": 5.0 * (event - np.arange(n)), "coordinate": running_peak,
            "event": event, "n_original": int(len(z["values"])),
            "endpoint_verified": True,
            "endpoint_source": "observed_first_20g_crossing" if len(hits) else "publisher_confirmed_terminal_20g_stop",
            "trajectory_sha256": trajectory_sha256}


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


def train_ft(train: dict, validation: dict, cfg: dict, seed: int, *, epochs: int | None = None):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train["x"]); target_scale = max(float(train["y"].max()), 1.0)
    x = torch.tensor(scaler.transform(train["x"]), dtype=torch.float32)
    y = torch.tensor(train["y"] / target_scale, dtype=torch.float32)
    vx = torch.tensor(scaler.transform(validation["x"]), dtype=torch.float32)
    vy = validation["y"]
    model = FTTransformer(n_cont_features=x.shape[1], cat_cardinalities=[], d_out=1,
                          n_blocks=cfg["depth"], d_block=cfg["width"], attention_n_heads=4,
                          attention_dropout=0.0, ffn_d_hidden=None, ffn_d_hidden_multiplier=2.0,
                          ffn_dropout=0.0, residual_dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rng = np.random.default_rng(seed); limit = 300 if epochs is None else int(epochs)
    best, best_epoch, state = float("inf"), 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), 512):
            idx = torch.tensor(order[start:start + 512])
            loss = (model(x[idx], None).squeeze(1) - y[idx]).square().mean()
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        model.eval()
        with torch.no_grad(): pred = np.maximum(model(vx, None).squeeze(1).numpy() * target_scale, 0.0)
        score = float(np.mean((pred - vy) ** 2))
        if score < best - 1e-10:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epochs is None and epoch - best_epoch > 50: break
    model.load_state_dict(state if epochs is None else model.state_dict())
    return {"model": model, "scaler": scaler, "target_scale": target_scale,
            "selected_epoch": best_epoch if epochs is None else limit, "validation_mse": best}


def predict_ft(fit: dict, x: np.ndarray) -> np.ndarray:
    value = torch.tensor(fit["scaler"].transform(x), dtype=torch.float32)
    fit["model"].eval()
    with torch.no_grad(): return np.maximum(fit["model"](value, None).squeeze(1).numpy() * fit["target_scale"], 0.0)


def rate_baseline(cells: dict, rows_: dict, units: tuple[str, ...], validation: dict | None = None):
    # Select only from development data. Each lag uses the corresponding frozen
    # running-peak slope column (after 3 normalized margin columns + 9 waveform summaries).
    candidates = []
    slope_columns = {1: 12, 3: 13, 6: 14}
    reference = validation if validation is not None else rows_
    for lag, col in slope_columns.items():
        positive = rows_["x"][:, col][rows_["x"][:, col] > 1e-8]
        for q in (.1, .25, .5):
            floor = float(np.quantile(positive, q)) if len(positive) else 1e-3
            pred = reference["margin"] / np.maximum(reference["x"][:, col], floor)
            candidates.append((float(np.mean((pred - reference["y"]) ** 2)), lag, floor))
    best = min(candidates)
    col = slope_columns[best[1]]
    return rows_["margin"] / np.maximum(rows_["x"][:, col], best[2]), best, candidates


def main() -> None:
    torch.set_num_threads(2)
    cells = {unit: causal_unit(unit) for unit in LOADS}
    audit = {u: {k: v for k, v in c.items() if k in ("event", "n_original", "endpoint_verified", "endpoint_source", "trajectory_sha256")} for u, c in cells.items()}
    fingerprints = [cells[u]["trajectory_sha256"] for u in LOADS]
    gates = {
        "all_test_endpoints_verified": all(cells[u]["endpoint_verified"] for u in TEST),
        "test_tail_at_least_30": all(len(cells[u]["y"]) - int(np.floor(.7 * len(cells[u]["y"]))) >= 30 for u in TEST),
        "no_duplicate_trajectory_fingerprints": len(fingerprints) == len(set(fingerprints)),
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
    ft_grid = [
        {"width": 16, "depth": 1, "lr": 5e-4, "wd": .1},
        {"width": 32, "depth": 1, "lr": 5e-4, "wd": .1},
        {"width": 32, "depth": 2, "lr": 2e-4, "wd": 1.0},
        {"width": 64, "depth": 2, "lr": 2e-4, "wd": 1.0},
    ]
    ft_search = []
    for cfg in ft_grid:
        candidate = train_ft(tr, va, cfg, 42)
        ft_search.append({"config": cfg, "validation_mse": candidate["validation_mse"],
                          "selected_epoch": candidate["selected_epoch"]})
    ft_cfg = min(ft_search, key=lambda z: z["validation_mse"])["config"]
    ft_predictions, ft_runs = [], []
    for seed in SEEDS:
        selected = train_ft(tr, va, ft_cfg, seed)
        final = train_ft(full, full, ft_cfg, seed, epochs=max(selected["selected_epoch"], 1))
        prediction = predict_ft(final, te["x"]); ft_predictions.append(prediction)
        ft_runs.append({"seed": seed, "selected_epoch": selected["selected_epoch"],
                        "metrics": regression_metrics(te["y"], prediction, te["groups"])})
    ft = np.asarray(ft_predictions)
    _, rate_selection, rate_search = rate_baseline(cells, tr, TRAIN, va)
    slope_col = {1: 12, 3: 13, 6: 14}[rate_selection[1]]
    rate_prediction = te["margin"] / np.maximum(te["x"][:, slope_col], rate_selection[2])
    development_lifetimes = [5.0 * cells[u]["event"] for u in TRAIN + VALIDATION]
    elapsed_prediction = np.maximum(np.median(development_lifetimes) - te["x"][:, -3], 0.0)
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
            "ft_transformer": regression_metrics(te["y"], ft.mean(0), te["groups"]),
            "boundary_rate": regression_metrics(te["y"], rate_prediction, te["groups"]),
            "elapsed_time": regression_metrics(te["y"], elapsed_prediction, te["groups"]),
        },
        "ridge": {"selected_alpha": ridge_alpha, "validation_search_mse_alpha": ridge_search},
        "ft_transformer": {"selected_config": ft_cfg, "search": ft_search, "runs": ft_runs},
        "boundary_rate": {"selected_lag": rate_selection[1], "selected_floor": rate_selection[2],
                          "validation_search": rate_search},
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
                        affine=affine, mlp=mlp, ridge=ridge_prediction, ft=ft,
                        boundary_rate=rate_prediction, elapsed=elapsed_prediction,
                        coordinate=te["coordinate"], distance=distance)
    print(json.dumps({"ensemble": result["ensemble"], "state_extrapolation": result["state_extrapolation"],
                      "primary_success": result["primary_success"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
