#!/usr/bin/env python3
"""Frozen PP-X v1.0 on the SNL BatteryLife processed cohort.

Follows protocols/SNL_PPX_V1_PROTOCOL.md.  Used only because UL-PUR was
inconclusive for sample size.  Archive SHA-256 is written before pickle
parsing.  Test labels are used only after executor and hyperparameter
selection on validation RMSE.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, regression_metrics, select_affine_initialization
from pp_extrapolation.executor_policy import ExecutorEvidence, select_residual_executor

DATA = ROOT / "data" / "snl_ppx" / "SNL.zip"
OUT = ROOT / "results" / "snl_ppx_v1"
MANIFEST = ROOT / "data" / "snl_ppx" / "sha256_manifest.json"
SEEDS = (42, 43, 44, 45, 46)
MIN_DISCHARGE = 20
EOL_FRACTION = 0.80
HISTORY = 5
WIDTHS = (16, 32)
LEARNING_RATES = (5e-4, 1e-3)
EXECUTORS = {
    "unbounded": {"residual_decay": 0.0},
    "bounded": {"residual_decay": 0.3},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_series() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with zipfile.ZipFile(DATA) as archive:
        for name in sorted(item for item in archive.namelist() if item.endswith(".pkl")):
            record = pickle.loads(archive.read(name))
            cycles, capacities = [], []
            for row in record.get("cycle_data", []):
                values = np.asarray(row.get("discharge_capacity_in_Ah") or [], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0 or float(np.max(values)) <= 0:
                    continue
                cycles.append(float(row["cycle_number"]))
                capacities.append(float(np.max(values)))
            if not cycles:
                continue
            unique = np.unique(cycles)
            collapsed = [
                float(np.median(np.asarray(capacities)[np.asarray(cycles) == cycle]))
                for cycle in unique
            ]
            cell = str(record.get("cell_id") or Path(name).stem)
            ambient = record.get("temperature_in_C")
            if ambient is None:
                ambient = record.get("ambient_temperature")
            try:
                ambient = float(ambient)
            except (TypeError, ValueError):
                ambient = np.nan
            out[cell] = {
                "cycle": np.asarray(unique, dtype=np.float64),
                "capacity": np.asarray(collapsed, dtype=np.float64),
                "ambient": ambient,
            }
    return out


def eligibility(cell: str, series: dict) -> dict:
    capacity = series["capacity"]
    n = int(capacity.size)
    first5 = float(np.median(capacity[:5])) if n >= 5 else float("nan")
    threshold = EOL_FRACTION * first5 if np.isfinite(first5) else float("nan")
    crossing = None
    if np.isfinite(threshold):
        hits = np.flatnonzero(capacity <= threshold)
        if hits.size:
            crossing = int(hits[0])
    return {
        "cell": cell,
        "n_discharge": n,
        "first5_median": first5,
        "eol_threshold": threshold,
        "crossing_index": crossing,
        "eligible": bool(n >= MIN_DISCHARGE and crossing is not None and crossing >= HISTORY),
    }


def split_ids(eligible: list[str]) -> dict[str, list[str]]:
    n = len(eligible)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    return {
        "train": eligible[:n_train],
        "validation": eligible[n_train:n_train + n_val],
        "test": eligible[n_train + n_val:],
    }


def rows(library: dict[str, dict], names: list[str], crossings: dict[str, int],
         *, mode: str, boundary: float | None) -> dict:
    xs, ys, groups, health = [], [], [], []
    for name in names:
        capacity = library[name]["capacity"]
        cycle = library[name]["cycle"]
        ambient = library[name]["ambient"]
        stop = crossings[name]
        for index in range(HISTORY - 1, stop):
            window = capacity[index - HISTORY + 1:index + 1]
            value = float(capacity[index])
            rate = float(capacity[index] - capacity[index - 1])
            slope = float((window[-1] - window[0]) / max(HISTORY - 1, 1))
            if mode == "train" and boundary is not None and value < boundary:
                continue
            if mode != "train" and boundary is not None and value >= boundary:
                continue
            xs.append([
                value,
                rate,
                float(window.mean()),
                slope,
                float(cycle[index]),
                float(ambient) if np.isfinite(ambient) else 0.0,
            ])
            ys.append(float(stop - index))
            groups.append(name)
            health.append(value)
    if not xs:
        return {
            "x": np.zeros((0, 6), dtype=np.float32),
            "y": np.zeros((0,), dtype=np.float32),
            "groups": np.asarray([], dtype=object),
            "health": np.zeros((0,), dtype=np.float64),
        }
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "groups": np.asarray(groups),
        "health": np.asarray(health, dtype=np.float64),
    }


def bootstrap(y, pp, mlp, groups, replicates=20000):
    labels = np.unique(groups)
    index = {unit: np.flatnonzero(groups == unit) for unit in labels}
    rng = np.random.default_rng(20260910)
    gain = np.empty(replicates)
    for draw in range(replicates):
        chosen = rng.choice(labels, len(labels), replace=True)
        take = np.concatenate([index[unit] for unit in chosen])
        gain[draw] = (
            np.mean((mlp[take] - y[take]) ** 2) ** 0.5
            - np.mean((pp[take] - y[take]) ** 2) ** 0.5
        )
    return {
        "mean": float(gain.mean()),
        "95ci": [float(x) for x in np.quantile(gain, (0.025, 0.975))],
        "probability_positive": float(np.mean(gain > 0)),
        "replicates": replicates,
    }


def write(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    keep = (
        "status", "reason", "eligible_ids", "split_ids", "n",
        "executor_decision", "ppx_ensemble", "plain_mlp_ensemble",
        "confirmatory_success",
    )
    print(json.dumps({key: payload[key] for key in keep if key in payload}, indent=2), flush=True)


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError(f"Refusing to overwrite {OUT / 'results.json'}")

    digest = sha256(DATA)
    MANIFEST.write_text(json.dumps({
        "archive": "SNL.zip",
        "source": "https://zenodo.org/records/14934405/files/SNL.zip",
        "sha256": digest,
        "note": "Hash recorded before pickle parsing.",
    }, indent=2) + "\n")

    library = load_series()
    audits = [eligibility(cell, library[cell]) for cell in sorted(library)]
    eligible = [row["cell"] for row in audits if row["eligible"]]
    crossings = {row["cell"]: int(row["crossing_index"]) for row in audits if row["eligible"]}
    print("N_CELLS", len(library), "ELIGIBLE", len(eligible), flush=True)

    if len(eligible) < 10:
        write({
            "status": "inconclusive",
            "reason": "fewer than 10 eligible cells; IDs were not replaced",
            "model": "PP-X v1.0 core + validation-selected residual executor",
            "protocol": "SNL_PPX_V1",
            "archive_sha256": digest,
            "eligibility": audits,
            "eligible_ids": eligible,
        })
        return

    ids = split_ids(eligible)
    if len(ids["test"]) < 2:
        write({
            "status": "inconclusive",
            "reason": "fewer than 2 test cells after the frozen 60/20/20 split",
            "model": "PP-X v1.0 core + validation-selected residual executor",
            "protocol": "SNL_PPX_V1",
            "archive_sha256": digest,
            "eligibility": audits,
            "eligible_ids": eligible,
            "split_ids": ids,
        })
        return

    train_all = rows(library, ids["train"], crossings, mode="all", boundary=None)
    if len(train_all["y"]) == 0:
        write({
            "status": "infeasible",
            "reason": "no train rows before the observed 80% crossing",
            "eligibility": audits,
            "eligible_ids": eligible,
            "split_ids": ids,
        })
        return

    boundary = float(np.min(train_all["health"]))
    train = rows(library, ids["train"], crossings, mode="train", boundary=boundary)
    validation = rows(library, ids["validation"], crossings, mode="tail", boundary=boundary)
    test = rows(library, ids["test"], crossings, mode="tail", boundary=boundary)
    hull = {
        "train_min_capacity": boundary,
        "train_rows": int(len(train["y"])),
        "validation_rows": int(len(validation["y"])),
        "test_rows": int(len(test["y"])),
        "validation_scored_all_below": bool(len(validation["y"]) and np.all(validation["health"] < boundary)),
        "test_scored_all_below": bool(len(test["y"]) and np.all(test["health"] < boundary)),
    }
    print("HULL", json.dumps(hull), "SPLIT", ids, flush=True)

    if not (
        hull["validation_scored_all_below"]
        and hull["test_scored_all_below"]
        and len(train["y"]) > 0
        and len(validation["y"]) > 0
        and len(test["y"]) > 0
    ):
        write({
            "status": "infeasible",
            "reason": "a scored outer split is empty or not strictly below the actual train capacity boundary",
            "model": "PP-X v1.0 core + validation-selected residual executor",
            "protocol": "SNL_PPX_V1",
            "archive_sha256": digest,
            "eligibility": audits,
            "eligible_ids": eligible,
            "split_ids": ids,
            "hull": hull,
        })
        return

    affine = select_affine_initialization(train, validation)
    pp_search = []
    policy_rows = []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            losses = {}
            extras = {}
            for name, extra in EXECUTORS.items():
                fit = fit_pp(
                    train, validation, seed=42, affine_selection=affine,
                    max_epochs=300, patience=50, width=width, learning_rate=lr,
                    weight_decay=2.0, **extra,
                )
                loss = float(fit.selection["validation_mse"])
                losses[name] = loss
                extras[name] = extra
                pp_search.append({
                    "width": width, "learning_rate": lr, "executor": name,
                    "validation_mse": loss,
                    "selected_epoch": int(fit.selection["selected_epoch"]),
                })
                print("PP_VAL", width, lr, name, loss, flush=True)
            decision = select_residual_executor(ExecutorEvidence(
                unbounded_loss=losses["unbounded"],
                bounded_loss=losses["bounded"],
                support_heterogeneity=0.0,
            ))
            policy_rows.append({
                "width": width, "learning_rate": lr,
                "executor": decision.executor, "validation_mse": losses[decision.executor],
                "reason": decision.reason, "extra": extras[decision.executor],
            })
    chosen_pp = min(policy_rows, key=lambda row: row["validation_mse"])
    print("PP_SELECTED", chosen_pp, flush=True)

    mlp_search = []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            fit = fit_plain(
                train, validation, seed=42, max_epochs=300, patience=50,
                width=width, learning_rate=lr, weight_decay=2.0,
            )
            mlp_search.append({
                "width": width, "learning_rate": lr,
                "validation_mse": float(fit["validation_mse"]),
                "selected_epoch": int(fit["selected_epoch"]),
            })
            print("MLP_VAL", width, lr, fit["validation_mse"], flush=True)
    chosen_mlp = min(mlp_search, key=lambda row: row["validation_mse"])
    print("MLP_SELECTED", chosen_mlp, flush=True)

    pp_preds, mlp_preds = [], []
    pp_runs, mlp_runs = [], []
    for seed in SEEDS:
        pp_fit = fit_pp(
            train, validation, seed=seed, affine_selection=affine,
            max_epochs=300, patience=50, width=chosen_pp["width"],
            learning_rate=chosen_pp["learning_rate"], weight_decay=2.0,
            **chosen_pp["extra"],
        )
        mlp_fit = fit_plain(
            train, validation, seed=seed, max_epochs=300, patience=50,
            width=chosen_mlp["width"], learning_rate=chosen_mlp["learning_rate"],
            weight_decay=2.0,
        )
        pp_pred = predict(pp_fit, test["x"])
        mlp_pred = predict_plain(mlp_fit, test["x"])
        pp_preds.append(pp_pred)
        mlp_preds.append(mlp_pred)
        pp_scores = regression_metrics(test["y"], pp_pred, test["groups"])
        mlp_scores = regression_metrics(test["y"], mlp_pred, test["groups"])
        pp_runs.append({"seed": seed, **pp_scores, "selected_epoch": int(pp_fit.selection["selected_epoch"])})
        mlp_runs.append({"seed": seed, **mlp_scores, "selected_epoch": int(mlp_fit["selected_epoch"])})
        print("SEED", seed, "PP", pp_scores["pooled"]["r2"], "MLP", mlp_scores["pooled"]["r2"], flush=True)

    pp_matrix = np.asarray(pp_preds)
    mlp_matrix = np.asarray(mlp_preds)
    pp_ensemble = regression_metrics(test["y"], pp_matrix.mean(0), test["groups"])
    mlp_ensemble = regression_metrics(test["y"], mlp_matrix.mean(0), test["groups"])
    paired = bootstrap(test["y"], pp_matrix.mean(0), mlp_matrix.mean(0), test["groups"])
    success = bool(
        pp_ensemble["pooled"]["r2"] is not None
        and pp_ensemble["pooled"]["r2"] > 0
        and pp_ensemble["pooled"]["rmse"] < mlp_ensemble["pooled"]["rmse"]
    )
    np.savez_compressed(
        OUT / "predictions.npz",
        truth=test["y"], groups=test["groups"], ppx=pp_matrix, mlp=mlp_matrix,
        health=test["health"],
    )
    write({
        "status": "external confirmation complete",
        "model": "PP-X v1.0 core + validation-selected residual executor",
        "dataset": "SNL BatteryLife processed",
        "protocol": "SNL_PPX_V1",
        "archive_sha256": digest,
        "eligibility": audits,
        "eligible_ids": eligible,
        "split_ids": ids,
        "hull": hull,
        "n": {
            "eligible_cells": len(eligible),
            "train_cells": len(ids["train"]),
            "validation_cells": len(ids["validation"]),
            "test_cells": len(ids["test"]),
            "train_rows": int(len(train["y"])),
            "validation_rows": int(len(validation["y"])),
            "test_rows": int(len(test["y"])),
        },
        "pp_search": pp_search,
        "mlp_search": mlp_search,
        "executor_decision": {
            "executor": chosen_pp["executor"],
            "reason": chosen_pp["reason"],
            "width": chosen_pp["width"],
            "learning_rate": chosen_pp["learning_rate"],
        },
        "mlp_selection": chosen_mlp,
        "ppx_runs": pp_runs,
        "mlp_runs": mlp_runs,
        "ppx_ensemble": pp_ensemble,
        "plain_mlp_ensemble": mlp_ensemble,
        "paired_cell_bootstrap": paired,
        "confirmatory_success": success,
        "note": (
            "Model-level external cohort. Not a per-unit routing-gate confirmation "
            "if the test split has fewer than ten units."
        ),
    })


if __name__ == "__main__":
    main()
