#!/usr/bin/env python3
"""PP-X core + validation-selected residual executor on the locked Misata split.

Does not overwrite the original untouched PP confirmation.  Test labels are
used only after executor selection on train-machine validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from misata_machine_untouched import prepare, rows, metric, bootstrap, SEEDS
from pp_extrapolation import fit_pp, predict, select_affine_initialization
from pp_extrapolation.executor_policy import ExecutorEvidence, select_residual_executor

DATA = ROOT / "data" / "misata_machine" / "readings.csv"
OUT = ROOT / "results" / "misata_ppx_v1"
OLD = ROOT / "results" / "misata_machine_untouched_v1" / "predictions_frozen_before_scoring.npz"


def val_mse(fit, validation: dict) -> float:
    pred = predict(fit, validation["x"])
    return float(np.mean((pred - validation["y"]) ** 2))


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError(f"Refusing to overwrite {OUT / 'results.json'}")

    frame = pd.read_csv(DATA)
    units, modes, controls, max_length = prepare(frame)
    train_ids = sorted(str(x) for x in frame.loc[frame.split == "train", "unit_id"].unique())
    test_ids = sorted(str(x) for x in frame.loc[frame.split == "test", "unit_id"].unique())
    order = np.asarray(train_ids)[np.random.default_rng(42).permutation(len(train_ids))]
    fit_ids, val_ids = list(order[:64]), list(order[64:])

    train = rows(units, fit_ids, modes, controls, max_length, 0.30, stride=3)
    validation = rows(units, val_ids, modes, controls, max_length, 0.30)
    affine = select_affine_initialization(train, validation)

    executors = {
        "unbounded": {"residual_decay": 0.0},
        "bounded": {"residual_decay": 0.3},
    }
    search = []
    for name, extra in executors.items():
        fit = fit_pp(
            train, validation, seed=42, affine_selection=affine,
            max_epochs=300, patience=50, width=16, learning_rate=1e-3,
            weight_decay=2.0, **extra,
        )
        mse = val_mse(fit, validation)
        search.append({"executor": name, "validation_mse": mse, "extra": extra,
                       "selected_epoch": int(fit.selection["selected_epoch"])})
        print("VAL", name, mse, flush=True)

    by = {row["executor"]: row["validation_mse"] for row in search}
    decision = select_residual_executor(ExecutorEvidence(
        unbounded_loss=by["unbounded"],
        bounded_loss=by["bounded"],
        support_heterogeneity=0.0,
    ))
    chosen = next(row for row in search if row["executor"] == decision.executor)
    print("SELECTED", decision.executor, decision.reason, flush=True)

    full_train = rows(units, train_ids, modes, controls, max_length, 0.30, stride=3)
    test = rows(units, test_ids, modes, controls, max_length, 0.30)
    full_affine = select_affine_initialization(full_train, validation)
    preds = []
    runs = []
    for seed in SEEDS:
        fit = fit_pp(
            full_train, validation, seed=seed, affine_selection=full_affine,
            max_epochs=300, patience=50, width=16, learning_rate=1e-3,
            weight_decay=2.0, **chosen["extra"],
        )
        pred = predict(fit, test["x"])
        preds.append(pred)
        scores = metric(test["y"], pred)
        runs.append({"seed": seed, **scores, "selected_epoch": int(fit.selection["selected_epoch"])})
        print("SEED", seed, scores["r2"], flush=True)

    matrix = np.asarray(preds)
    ensemble = metric(test["y"], matrix.mean(0))
    mlp = None
    if OLD.exists():
        old = np.load(OLD, allow_pickle=True)
        if np.array_equal(np.asarray(old["truth"]), test["y"]):
            mlp = metric(test["y"], np.asarray(old["mlp"]).mean(0))
            paired = bootstrap(test["y"], matrix.mean(0), np.asarray(old["mlp"]).mean(0), test["groups"])
        else:
            paired = None
    else:
        paired = None

    np.savez_compressed(OUT / "predictions.npz", truth=test["y"], groups=test["groups"], ppx=matrix)
    payload = {
        "model": "PP-X v1.0 core + validation-selected residual executor",
        "dataset": "Misata machine-degradation locked publisher split",
        "n": {"train_machines": len(train_ids), "val_machines": len(val_ids),
              "test_machines": len(test_ids), "train_rows": len(full_train["y"]),
              "test_rows": len(test["y"])},
        "executor_search": search,
        "executor_decision": {"executor": decision.executor, "reason": decision.reason},
        "runs": runs,
        "ensemble": ensemble,
        "matched_mlp_from_locked_artifact": mlp,
        "paired_machine_bootstrap": paired,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("ENSEMBLE", ensemble, flush=True)


if __name__ == "__main__":
    main()
