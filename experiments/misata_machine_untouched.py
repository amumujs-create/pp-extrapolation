#!/usr/bin/env python3
"""Frozen compact synthetic machine-cohort confirmation for PP."""
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

DATA = ROOT / "data" / "misata_machine" / "readings.csv"
ARCHIVE = ROOT / "data" / "misata_machine" / "machine-degradation.zip"
OUT = ROOT / "results" / "misata_machine_untouched_v1"
SEEDS = (42, 43, 44, 45, 46)
SIGNALS = ("tool_wear_min", "vibration_mm_s", "torque_nm",
           "process_temperature_k", "air_temperature_k", "rotational_speed_rpm")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def prepare(frame):
    train = frame[frame["split"] == "train"]
    medians = train[list(SIGNALS)].median().to_numpy(float)
    modes = sorted(train["failure_mode"].unique())
    controls = sorted(train["control_type"].unique())
    max_length = int(train.groupby("unit_id").size().max())
    units = {}
    for unit, raw in frame.groupby("unit_id"):
        block = raw.sort_values("cycle").reset_index(drop=True)
        values = block[list(SIGNALS)].to_numpy(float)
        missing = ~np.isfinite(values)
        # Strictly causal forward fill, followed by publisher-train medians.
        values = pd.DataFrame(values).ffill().to_numpy(float)
        values = np.where(np.isfinite(values), values, medians)
        units[str(unit)] = {"frame": block, "values": values,
                            "missing": missing.astype(float)}
    return units, modes, controls, max_length


def at_feature(unit, endpoint, modes, controls, max_length):
    block, values, missing = unit["frame"], unit["values"], unit["missing"]
    prefix = values[:endpoint + 1]
    recent = prefix[-min(20, len(prefix)):]
    t = np.arange(len(recent), dtype=float); tc = t - t.mean()
    slopes = tc @ recent / max(float(tc @ tc), 1.0)
    curvature = np.mean(np.diff(recent, 2, axis=0), axis=0) if len(recent) >= 3 else np.zeros(len(SIGNALS))
    current = values[endpoint]
    miss_now = missing[endpoint]
    mode = block["failure_mode"].iloc[endpoint]
    control = block["control_type"].iloc[endpoint]
    cycle = float(block["cycle"].iloc[endpoint])
    return np.concatenate((
        current, recent.mean(0), recent.std(0), slopes, curvature, miss_now,
        [np.log1p(cycle), min((endpoint + 1) / max_length, 1.0)],
        [float(mode == x) for x in modes], [float(control == x) for x in controls],
    )).astype(np.float32)


def rows(units, names, modes, controls, max_length, fraction, *, stride=1):
    xs, ys, groups = [], [], []
    for name in names:
        unit = units[name]; n = len(unit["frame"])
        start = max(1, int(np.floor((1.0 - fraction) * (n - 1))))
        endpoints = list(range(start, n - 1, stride))
        if endpoints and endpoints[-1] != n - 2: endpoints.append(n - 2)
        for endpoint in endpoints:
            xs.append(at_feature(unit, endpoint, modes, controls, max_length))
            ys.append(float(unit["frame"]["rul_cycles"].iloc[endpoint]))
            groups.append(name)
    return {"x": np.asarray(xs), "y": np.asarray(ys), "groups": np.asarray(groups)}


def metric(y, p):
    return {"r2": float(r2_score(y, p)),
            "rmse": float(mean_squared_error(y, p) ** 0.5),
            "mae": float(mean_absolute_error(y, p))}


def tune(units, fit_units, val_units, modes, controls, max_length, model):
    common = itertools.product((16, 32, 64), (5e-4, 1e-3), (0.5, 2.0, 5.0))
    candidates = []
    for fraction in (0.30, 0.50, 0.70):
        train = rows(units, fit_units, modes, controls, max_length, fraction, stride=3)
        validation = rows(units, val_units, modes, controls, max_length, 0.30)
        affine = select_affine_initialization(train, validation) if model == "pp" else None
        for width, learning_rate, weight_decay in list(common):
            extras = ({}, {"learned_affine_gate": True,
                           "direct_residual_mixture": True}) if model == "pp" else ({},)
            for extra in extras:
                config = {"width": width, "learning_rate": learning_rate,
                          "weight_decay": weight_decay, **extra}
                if model == "pp":
                    fit = fit_pp(train, validation, seed=42, affine_selection=affine,
                                 max_epochs=300, patience=50, **config)
                    pred = predict(fit, validation["x"]); epoch = fit.selection["selected_epoch"]
                else:
                    fit = fit_plain(train, validation, seed=42, max_epochs=300,
                                    patience=50, **config)
                    pred = predict_plain(fit, validation["x"]); epoch = fit["selected_epoch"]
                scores = [np.mean((pred[validation["groups"] == u]
                                  - validation["y"][validation["groups"] == u]) ** 2) ** 0.5
                          for u in val_units]
                candidates.append({"fraction": fraction, "config": config,
                                   "stage1_rmse": float(np.mean(scores)),
                                   "stage1_epoch": int(epoch)})
        common = itertools.product((16, 32, 64), (5e-4, 1e-3), (0.5, 2.0, 5.0))
    candidates.sort(key=lambda x: x["stage1_rmse"]); finalists = candidates[:5]
    for row in finalists:
        train = rows(units, fit_units, modes, controls, max_length, row["fraction"], stride=3)
        validation = rows(units, val_units, modes, controls, max_length, 0.30)
        affine = select_affine_initialization(train, validation) if model == "pp" else None
        per_seed, epochs = [], []
        for seed in (42, 43, 44):
            if model == "pp":
                fit = fit_pp(train, validation, seed=seed, affine_selection=affine,
                             max_epochs=450, patience=70, **row["config"])
                pred = predict(fit, validation["x"]); epochs.append(fit.selection["selected_epoch"])
            else:
                fit = fit_plain(train, validation, seed=seed, max_epochs=450,
                                patience=70, **row["config"])
                pred = predict_plain(fit, validation["x"]); epochs.append(fit["selected_epoch"])
            per_seed.append(np.mean([np.mean((pred[validation["groups"] == u]
                                - validation["y"][validation["groups"] == u]) ** 2) ** 0.5
                                     for u in val_units]))
        row["stage2_rmse"] = float(np.mean(per_seed))
        row["selected_epoch"] = int(max(1, round(np.median(epochs))))
    best = min(x["stage2_rmse"] for x in finalists)
    near = [x for x in finalists if x["stage2_rmse"] <= 1.01 * best]
    selected = min(near, key=lambda x: (x["config"]["width"],
                    -x["config"]["weight_decay"], x["stage2_rmse"]))
    return selected, finalists


def final(units, train_units, test_units, modes, controls, max_length, selected, model):
    train = rows(units, train_units, modes, controls, max_length,
                 selected["fraction"], stride=3)
    test = rows(units, test_units, modes, controls, max_length, 0.30)
    predictions, epochs = [], []
    affine = select_affine_initialization(train, train) if model == "pp" else None
    for seed in SEEDS:
        if model == "pp":
            fit = fit_pp(train, train, seed=seed, affine_selection=affine,
                         max_epochs=selected["selected_epoch"],
                         patience=selected["selected_epoch"] + 1, **selected["config"])
            predictions.append(predict(fit, test["x"])); epochs.append(fit.selection["selected_epoch"])
        else:
            fit = fit_plain(train, train, seed=seed, max_epochs=selected["selected_epoch"],
                            patience=selected["selected_epoch"] + 1,
                            restore_best=False, **selected["config"])
            predictions.append(predict_plain(fit, test["x"])); epochs.append(fit["selected_epoch"])
    return test, np.asarray(predictions), epochs


def bootstrap(y, pp, mlp, groups, n=20000):
    labels = np.unique(groups); ix = {u: np.flatnonzero(groups == u) for u in labels}
    rng = np.random.default_rng(20260909); gain = np.empty(n)
    for b in range(n):
        chosen = rng.choice(labels, len(labels), replace=True)
        index = np.concatenate([ix[u] for u in chosen])
        gain[b] = np.mean((mlp[index] - y[index]) ** 2) ** 0.5 - np.mean((pp[index] - y[index]) ** 2) ** 0.5
    return {"mean": float(gain.mean()), "95ci": [float(x) for x in np.quantile(gain, (.025, .975))],
            "probability_positive": float(np.mean(gain > 0)), "replicates": n}


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA); units, modes, controls, max_length = prepare(frame)
    train_ids = sorted(str(x) for x in frame.loc[frame.split == "train", "unit_id"].unique())
    test_ids = sorted(str(x) for x in frame.loc[frame.split == "test", "unit_id"].unique())
    order = np.asarray(train_ids)[np.random.default_rng(42).permutation(len(train_ids))]
    fit_ids, val_ids = list(order[:64]), list(order[64:])
    pp_selected, pp_search = tune(units, fit_ids, val_ids, modes, controls, max_length, "pp")
    print("PP_SELECTED", pp_selected, flush=True)
    mlp_selected, mlp_search = tune(units, fit_ids, val_ids, modes, controls, max_length, "mlp")
    print("MLP_SELECTED", mlp_selected, flush=True)
    test, pp_runs, pp_epochs = final(units, train_ids, test_ids, modes, controls, max_length, pp_selected, "pp")
    _, mlp_runs, mlp_epochs = final(units, train_ids, test_ids, modes, controls, max_length, mlp_selected, "mlp")
    np.savez_compressed(OUT / "predictions_frozen_before_scoring.npz",
                        pp=pp_runs, mlp=mlp_runs, truth=test["y"], groups=test["groups"])
    y, pp, mlp = test["y"], pp_runs.mean(0), mlp_runs.mean(0)
    inference = bootstrap(y, pp, mlp, test["groups"])
    result = {"status": "untouched controlled-synthetic external confirmation complete",
              "protocol_commit": "23f3827", "archive_sha256": sha256(ARCHIVE),
              "n": {"train_machines": len(train_ids), "test_machines": len(test_ids),
                    "test_rows": len(y)}, "pp_selection": pp_selected,
              "pp_finalists": pp_search, "mlp_selection": mlp_selected,
              "mlp_finalists": mlp_search, "pp_epochs": pp_epochs, "mlp_epochs": mlp_epochs,
              "pp": metric(y, pp), "plain_mlp": metric(y, mlp),
              "paired_machine_bootstrap": inference}
    result["confirmatory_success"] = bool(result["pp"]["r2"] > 0
        and result["pp"]["rmse"] < result["plain_mlp"]["rmse"] and inference["95ci"][0] > 0)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("pp", "plain_mlp", "paired_machine_bootstrap", "confirmatory_success")}, indent=2), flush=True)


if __name__ == "__main__": main()
