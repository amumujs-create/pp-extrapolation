#!/usr/bin/env python3
"""Posthoc repair of the FD001/FD003 PP representation.

The model remains one frozen-affine-plus-neural-residual PP.  The change is a
train-only operating-condition normalization and causal initial-state delta,
which prevent operating offsets from being mistaken for degradation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from cmapss_single_condition import (
    RUL_CAP, SingleConditionNormalizer, load_fd, make_rows, split_table_units,
)
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, select_affine_initialization

SEEDS = (42, 43, 44, 45, 46)
OUT = ROOT / "results" / "cmapss_regime_normalized_pp_v1"


def as_pp(rows: dict) -> dict:
    return {
        "x": np.asarray(rows["x"]),
        "y": np.asarray(rows["y"]),
        "groups": np.asarray(rows["units"]).astype(str),
    }


def prepare(fd: str):
    train_table, test_table, truth = load_fd(fd)
    fit_table, validation_table, fit_units, validation_units = split_table_units(
        train_table, seed=42, validation_fraction=0.2
    )
    selection_normalizer = SingleConditionNormalizer.fit(fit_table)
    train = as_pp(make_rows(fit_table, selection_normalizer, stride=5))
    validation = as_pp(make_rows(validation_table, selection_normalizer, stride=5))

    final_normalizer = SingleConditionNormalizer.fit(train_table)
    full = as_pp(make_rows(train_table, final_normalizer, stride=5))
    test_x = make_rows(test_table, final_normalizer, final_only=True)["x"]
    return train, validation, full, test_x, np.asarray(truth), fit_units, validation_units


def metrics(y, p):
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "mae": float(mean_absolute_error(y, p)),
    }


def paired_inference(y, pp, mlp, *, seed: int) -> dict:
    """Engine-paired uncertainty for the ensemble prediction comparison."""
    rng = np.random.default_rng(seed)
    n = len(y)
    index = rng.integers(0, n, size=(20_000, n))
    delta = (
        np.sqrt(np.mean((mlp[index] - y[index]) ** 2, axis=1))
        - np.sqrt(np.mean((pp[index] - y[index]) ** 2, axis=1))
    )
    loss_delta = (mlp - y) ** 2 - (pp - y) ** 2
    signs = rng.choice((-1.0, 1.0), size=(100_000, n))
    null = np.mean(signs * loss_delta, axis=1)
    observed = float(np.mean(loss_delta))
    return {
        "rmse_gain_mlp_minus_pp": float(
            mean_squared_error(y, mlp) ** 0.5 - mean_squared_error(y, pp) ** 0.5
        ),
        "paired_engine_bootstrap_95ci": [float(x) for x in np.quantile(delta, (0.025, 0.975))],
        "bootstrap_probability_positive": float(np.mean(delta > 0)),
        "one_sided_paired_sign_flip_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "engine_absolute_error_win_fraction": float(np.mean(np.abs(pp - y) < np.abs(mlp - y))),
        "bootstrap_replicates": 20_000,
        "permutation_replicates": 100_000,
    }


def run_subset(fd: str) -> dict:
    train, validation, full, test_x, truth, fit_units, validation_units = prepare(fd)
    affine_selection = select_affine_initialization(train, validation)
    pp_predictions, mlp_predictions, runs = [], [], []
    for seed in SEEDS:
        selected_pp = fit_pp(
            train, validation, seed=seed, affine_selection=affine_selection,
            max_epochs=400, patience=70,
        )
        pp_epoch = max(int(selected_pp.selection["selected_epoch"]), 1)
        final_affine = select_affine_initialization(
            full, full, alphas=(affine_selection["selected_alpha"],)
        )
        final_pp = fit_pp(
            full, full, seed=seed, affine_selection=final_affine,
            max_epochs=pp_epoch, patience=pp_epoch + 1,
        )

        selected_mlp = fit_plain(train, validation, seed=seed, max_epochs=400, patience=70)
        mlp_epoch = max(int(selected_mlp["selected_epoch"]), 1)
        final_mlp = fit_plain(
            full, full, seed=seed, max_epochs=mlp_epoch,
            patience=mlp_epoch + 1, restore_best=False,
        )
        pp_prediction = predict(final_pp, test_x)
        mlp_prediction = predict_plain(final_mlp, test_x)
        pp_predictions.append(pp_prediction)
        mlp_predictions.append(mlp_prediction)
        runs.append({
            "seed": seed,
            "pp_epoch": pp_epoch,
            "mlp_epoch": mlp_epoch,
            "pp": metrics(truth, pp_prediction),
            "plain_mlp": metrics(truth, mlp_prediction),
        })
        print(fd, seed, runs[-1]["pp"]["r2"], runs[-1]["plain_mlp"]["r2"], flush=True)

    pp = np.asarray(pp_predictions)
    mlp = np.asarray(mlp_predictions)
    result = {
        "n_train_units": int(len(fit_units)),
        "n_validation_units": int(len(validation_units)),
        "n_test_units": int(len(truth)),
        "target": (
            f"official endpoint RUL; training labels and predictions capped at {RUL_CAP:g}, "
            "official test truth unchanged"
        ),
        "runs": runs,
        "ensemble": {
            "pp": metrics(truth, pp.mean(0)),
            "plain_mlp": metrics(truth, mlp.mean(0)),
        },
        "seed_mean_r2": {
            "pp": float(np.mean([r["pp"]["r2"] for r in runs])),
            "plain_mlp": float(np.mean([r["plain_mlp"]["r2"] for r in runs])),
        },
        "paired_inference": paired_inference(
            truth, pp.mean(0), mlp.mean(0), seed=20260909 + int(fd[-1])
        ),
    }
    result["pp_beats_mlp"] = bool(
        result["ensemble"]["pp"]["rmse"] < result["ensemble"]["plain_mlp"]["rmse"]
    )
    np.savez_compressed(OUT / f"{fd.lower()}_predictions.npz", y=truth, pp=pp, mlp=mlp)
    return result


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    subsets = {fd: run_subset(fd) for fd in ("FD001", "FD003")}
    payload = {
        "status": "posthoc representation repair; not untouched confirmation",
        "estimand": "unseen-engine official endpoint RUL with standard 125-cycle capped training target",
        "model": "one PP network: frozen affine path plus learned tanh residual",
        "representation": (
            "train-only operating-condition normalization; 30-cycle causal last/mean/std/slope; "
            "first-20-cycle sensor delta; current operating condition and log cycle"
        ),
        "test_used_for_epoch_or_hyperparameter_selection": False,
        "subsets": subsets,
        "combined_success": all(x["pp_beats_mlp"] for x in subsets.values()),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({fd: x["ensemble"] for fd, x in subsets.items()}, indent=2))


if __name__ == "__main__":
    main()
