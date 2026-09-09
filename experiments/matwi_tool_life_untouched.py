#!/usr/bin/env python3
"""Preregistered small-cohort confirmation on MATWI tool-life labels."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, select_affine_initialization

DATA = ROOT / "data" / "matwi" / "labels.csv"
OUT = ROOT / "results" / "matwi_tool_life_untouched_v1"
SEEDS = (42, 43, 44, 45, 46)
TYPES = ("flank_wear", "adhesion", "both")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_tools(frame):
    tools, audit = {}, {}
    for unit, raw in frame.groupby("Set"):
        raw = raw.copy()
        finite_id = raw["ImageID"].notna()
        block = raw.loc[finite_id].sort_values(["ImageID", "ImageDateTime"])
        duplicated = block.duplicated("ImageID", keep="first")
        block = block.loc[~duplicated].copy()
        finite_fraction = float(block["wear"].notna().mean()) if len(block) else 0.0
        eligible = len(block) >= 30 and finite_fraction >= 0.80
        if eligible:
            block["wear"] = block["wear"].ffill()
            block["type"] = block["type"].ffill()
            block = block.loc[block["wear"].notna()].reset_index(drop=True)
            eligible = len(block) >= 30
        audit[str(int(unit))] = {
            "raw_rows": int(len(raw)), "ordered_image_rows": int(len(block)),
            "missing_image_id": int((~finite_id).sum()),
            "duplicate_image_id": int(duplicated.sum()),
            "finite_wear_fraction": finite_fraction, "eligible": bool(eligible),
        }
        if eligible:
            tools[str(int(unit))] = block
    return tools, audit


def feature(block, endpoint, max_development_length):
    wear = block["wear"].to_numpy(float)[: endpoint + 1] / 100.0
    window = wear[-min(10, len(wear)):]
    t = np.arange(len(window), dtype=float)
    tc = t - t.mean()
    slope = float(tc @ window / max(float(tc @ tc), 1.0))
    differences = np.diff(wear)
    robust = float(np.median(differences)) if len(differences) else 0.0
    curvature = float(np.mean(np.diff(window, 2))) if len(window) >= 3 else 0.0
    fitted = window.mean() + slope * tc
    noise = float(np.sqrt(np.mean((window - fitted) ** 2)))
    current_type = str(block["type"].iloc[endpoint])
    one_hot = [float(current_type == name) for name in TYPES]
    prefix = endpoint + 1
    return np.asarray([
        wear[-1], wear[-1] - wear[0], window.mean(), window.std(), slope,
        robust, curvature, noise, np.log1p(prefix),
        min(prefix / max_development_length, 1.0), *one_hot,
    ], dtype=np.float32)


def make_rows(tools, units, *, tail_fraction, max_development_length, with_y=True):
    xs, ys, groups = [], [], []
    for unit in units:
        block = tools[unit]
        n = len(block)
        first = max(1, int(np.floor((1.0 - tail_fraction) * (n - 1))))
        for endpoint in range(first, n - 1):
            xs.append(feature(block, endpoint, max_development_length))
            if with_y:
                ys.append(float(n - 1 - endpoint))
            groups.append(unit)
    rows = {"x": np.asarray(xs), "groups": np.asarray(groups)}
    if with_y:
        rows["y"] = np.asarray(ys, dtype=np.float64)
    return rows


def metric(y, prediction):
    return {
        "r2": float(r2_score(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def folds(units):
    order = np.asarray(sorted(units))
    order = order[np.random.default_rng(2025).permutation(len(order))]
    return [list(x) for x in np.array_split(order, 4)]


def subset_tools(tools, units):
    return {u: tools[u] for u in units}


def evaluate_config(tools, dev_units, fraction, model, config, seeds, max_len):
    scores, epochs = [], []
    for held_out in folds(dev_units):
        train_units = [u for u in dev_units if u not in held_out]
        train = make_rows(tools, train_units, tail_fraction=fraction,
                          max_development_length=max_len)
        validation = make_rows(tools, held_out, tail_fraction=0.30,
                               max_development_length=max_len)
        affine = select_affine_initialization(train, validation) if model == "pp" else None
        for seed in seeds:
            if model == "pp":
                fit = fit_pp(train, validation, seed=seed, affine_selection=affine,
                             max_epochs=450 if len(seeds) > 1 else 300,
                             patience=60, **config)
                estimate = predict(fit, validation["x"])
                epoch = fit.selection["selected_epoch"]
            else:
                fit = fit_plain(train, validation, seed=seed,
                                max_epochs=450 if len(seeds) > 1 else 300,
                                patience=60, **config)
                estimate = predict_plain(fit, validation["x"])
                epoch = fit["selected_epoch"]
            for unit in held_out:
                mask = validation["groups"] == unit
                scores.append(float(np.mean((estimate[mask] - validation["y"][mask]) ** 2) ** 0.5))
            epochs.append(int(epoch))
    return float(np.mean(scores)), int(max(1, round(np.median(epochs))))


def tune(tools, dev_units, model, max_len):
    fractions = (0.30, 0.50, 0.70)
    common = list(itertools.product((16, 32, 64), (5e-4, 1e-3), (0.5, 2.0, 5.0)))
    candidates = []
    for fraction in fractions:
        for width, learning_rate, weight_decay in common:
            bases = ({}, {"learned_affine_gate": True, "direct_residual_mixture": True}) if model == "pp" else ({},)
            for extra in bases:
                config = {"width": width, "learning_rate": learning_rate,
                          "weight_decay": weight_decay, **extra}
                rmse, epoch = evaluate_config(
                    tools, dev_units, fraction, model, config, (42,), max_len
                )
                candidates.append({"fraction": fraction, "config": config,
                                   "stage1_rmse": rmse, "stage1_epoch": epoch})
    candidates.sort(key=lambda row: row["stage1_rmse"])
    finalists = candidates[:5]
    for row in finalists:
        rmse, epoch = evaluate_config(
            tools, dev_units, row["fraction"], model, row["config"],
            (42, 43, 44), max_len,
        )
        row["stage2_rmse"] = rmse
        row["selected_epoch"] = epoch
    best_rmse = min(row["stage2_rmse"] for row in finalists)
    near = [row for row in finalists if row["stage2_rmse"] <= best_rmse * 1.01]
    selected = min(near, key=lambda row: (
        row["config"]["width"], -row["config"]["weight_decay"],
        row["stage2_rmse"],
    ))
    return selected, finalists


def final_predictions(tools, dev_units, test_units, selected, model, max_len):
    train = make_rows(tools, dev_units, tail_fraction=selected["fraction"],
                      max_development_length=max_len)
    test = make_rows(tools, test_units, tail_fraction=0.30,
                     max_development_length=max_len)
    predictions, epochs = [], []
    if model == "pp":
        affine = select_affine_initialization(train, train)
    for seed in SEEDS:
        if model == "pp":
            fit = fit_pp(train, train, seed=seed, affine_selection=affine,
                         max_epochs=selected["selected_epoch"],
                         patience=selected["selected_epoch"] + 1,
                         **selected["config"])
            prediction = predict(fit, test["x"])
            epoch = fit.selection["selected_epoch"]
        else:
            fit = fit_plain(train, train, seed=seed,
                            max_epochs=selected["selected_epoch"],
                            patience=selected["selected_epoch"] + 1,
                            restore_best=False, **selected["config"])
            prediction = predict_plain(fit, test["x"])
            epoch = fit["selected_epoch"]
        predictions.append(prediction); epochs.append(int(epoch))
    return test, np.asarray(predictions), epochs


def bootstrap(y, pp, mlp, groups, replicates=20000):
    units = np.unique(groups); rng = np.random.default_rng(20260909)
    index = {u: np.flatnonzero(groups == u) for u in units}; gains = []
    for _ in range(replicates):
        chosen = rng.choice(units, len(units), replace=True)
        ix = np.concatenate([index[u] for u in chosen])
        gains.append(np.mean((mlp[ix] - y[ix]) ** 2) ** 0.5
                     - np.mean((pp[ix] - y[ix]) ** 2) ** 0.5)
    gains = np.asarray(gains)
    return {"mean": float(gains.mean()),
            "95ci": [float(x) for x in np.quantile(gains, (0.025, 0.975))],
            "probability_positive": float(np.mean(gains > 0)),
            "replicates": replicates}


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA); tools, audit = clean_tools(frame)
    identifiers = sorted(str(int(x)) for x in frame["Set"].unique())
    permutation = np.asarray(identifiers)[np.random.default_rng(42).permutation(len(identifiers))]
    dev_units = [u for u in permutation[:12] if u in tools]
    test_units = [u for u in permutation[12:] if u in tools]
    if len(test_units) < 4:
        raise RuntimeError("preregistered MATWI admissibility failed")
    max_len = float(max(len(tools[u]) for u in dev_units))
    pp_selected, pp_search = tune(tools, dev_units, "pp", max_len)
    print("PP_SELECTED", pp_selected, flush=True)
    mlp_selected, mlp_search = tune(tools, dev_units, "mlp", max_len)
    print("MLP_SELECTED", mlp_selected, flush=True)
    test, pp_runs, pp_epochs = final_predictions(
        tools, dev_units, test_units, pp_selected, "pp", max_len
    )
    _, mlp_runs, mlp_epochs = final_predictions(
        tools, dev_units, test_units, mlp_selected, "mlp", max_len
    )
    # Prediction artifact is written before any metric function is called.
    np.savez_compressed(OUT / "predictions_frozen_before_scoring.npz",
                        pp=pp_runs, mlp=mlp_runs, truth=test["y"],
                        groups=test["groups"], test_units=np.asarray(test_units))
    pp = pp_runs.mean(0); mlp = mlp_runs.mean(0); y = test["y"]
    inference = bootstrap(y, pp, mlp, test["groups"])
    result = {
        "status": "untouched external confirmation complete",
        "protocol_commit": "f5fc5e4",
        "dataset_doi": "10.48804/GK6LHH",
        "labels_sha256": sha256(DATA), "audit": audit,
        "split": {"development": dev_units, "test": test_units,
                  "ineligible_test": [u for u in permutation[12:] if u not in tools]},
        "n_scored_rows": len(y),
        "pp_selection": pp_selected, "pp_finalists": pp_search,
        "mlp_selection": mlp_selected, "mlp_finalists": mlp_search,
        "pp_epochs": pp_epochs, "mlp_epochs": mlp_epochs,
        "pp": metric(y, pp), "plain_mlp": metric(y, mlp),
        "per_test_tool": {
            u: {"pp": metric(y[test["groups"] == u], pp[test["groups"] == u]),
                "plain_mlp": metric(y[test["groups"] == u], mlp[test["groups"] == u])}
            for u in test_units
        },
        "paired_tool_bootstrap": inference,
    }
    result["confirmatory_success"] = bool(
        result["pp"]["r2"] > 0
        and result["pp"]["rmse"] < result["plain_mlp"]["rmse"]
        and inference["95ci"][0] > 0
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "pp", "plain_mlp", "paired_tool_bootstrap", "confirmatory_success"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
