#!/usr/bin/env python3
"""Frozen PP-X v1.0 on the NASA PCoE second battery cohort.

Follows protocols/NASA_PCOE_SECOND_COHORT_V1.md.  Does not overwrite the
B0005/6/7/18 development cell or any other NASA artifact.  Test labels are
used only after executor and hyperparameter selection on validation RMSE.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, regression_metrics, select_affine_initialization
from pp_extrapolation.executor_policy import ExecutorEvidence, select_residual_executor

DATA = ROOT / "data" / "nasa_pcoe_second"
OUT = ROOT / "results" / "nasa_pcoe_second_ppx_v1"
CANDIDATES = (
    "B0029", "B0030", "B0031", "B0032",
    "B0046", "B0047", "B0048",
    "B0053", "B0054", "B0055", "B0056",
)
SEEDS = (42, 43, 44, 45, 46)
MIN_DISCHARGE = 40
EOL_FRACTION = 0.70
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


def discharge_series(path: Path, cell: str) -> dict:
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    key = next(name for name in mat if not name.startswith("__"))
    cycles = np.atleast_1d(mat[key].cycle)
    capacities, ambients, currents = [], [], []
    for cycle in cycles:
        if str(getattr(cycle, "type", "")).lower() != "discharge":
            continue
        data = cycle.data
        if not hasattr(data, "Capacity"):
            continue
        capacity = np.asarray(data.Capacity, dtype=np.float64).reshape(-1)
        if capacity.size == 0:
            continue
        value = float(capacity[0])
        if not np.isfinite(value) or value <= 0:
            continue
        current = np.asarray(data.Current_measured, dtype=np.float64).ravel()
        ambient = float(np.asarray(cycle.ambient_temperature).reshape(-1)[0])
        capacities.append(value)
        ambients.append(ambient)
        currents.append(float(np.nanmean(np.abs(current))) if current.size else np.nan)
    return {
        "cell": cell,
        "capacity": np.asarray(capacities, dtype=np.float64),
        "ambient": np.asarray(ambients, dtype=np.float64),
        "current": np.asarray(currents, dtype=np.float64),
    }


def eligibility(series: dict) -> dict:
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
        "cell": series["cell"],
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
        ambient = library[name]["ambient"]
        current = library[name]["current"]
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
                float(index + 1),
                float(ambient[index]),
                float(current[index]) if np.isfinite(current[index]) else 0.0,
            ])
            ys.append(float(stop - index))
            groups.append(name)
            health.append(value)
    if not xs:
        return {
            "x": np.zeros((0, 7), dtype=np.float32),
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
    print(json.dumps({k: payload[k] for k in payload if k in (
        "status", "eligible_ids", "split_ids", "n", "executor_decision",
        "ppx_ensemble", "plain_mlp_ensemble", "confirmatory_success",
        "reason",
    )}, indent=2), flush=True)


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError(f"Refusing to overwrite {OUT / 'results.json'}")

    manifest = {name: sha256(DATA / f"{name}.mat") for name in CANDIDATES}
    (DATA / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    library = {name: discharge_series(DATA / f"{name}.mat", name) for name in CANDIDATES}
    audits = [eligibility(library[name]) for name in CANDIDATES]
    eligible = [row["cell"] for row in audits if row["eligible"]]
    crossings = {row["cell"]: int(row["crossing_index"]) for row in audits if row["eligible"]}
    print("ELIGIBILITY", json.dumps(audits, indent=2), flush=True)

    if len(eligible) < 10:
        write({
            "status": "inconclusive",
            "reason": "fewer than 10 eligible cells; IDs were not replaced",
            "model": "PP-X v1.0 core + validation-selected residual executor",
            "protocol": "NASA_PCOE_SECOND_COHORT_V1",
            "file_sha256": manifest,
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
            "protocol": "NASA_PCOE_SECOND_COHORT_V1",
            "file_sha256": manifest,
            "eligibility": audits,
            "eligible_ids": eligible,
            "split_ids": ids,
        })
        return

    train_all = rows(library, ids["train"], crossings, mode="all", boundary=None)
    if len(train_all["y"]) == 0:
        write({
            "status": "infeasible",
            "reason": "no train rows before the observed 70% crossing",
            "eligibility": audits,
            "eligible_ids": eligible,
            "split_ids": ids,
        })
        return

    boundary = float(np.min(train_all["health"]))
    train = rows(library, ids["train"], crossings, mode="train", boundary=boundary)
    validation = rows(library, ids["validation"], crossings, mode="tail", boundary=boundary)
    test = rows(library, ids["test"], crossings, mode="tail", boundary=boundary)
    val_all = rows(library, ids["validation"], crossings, mode="all", boundary=None)
    test_all = rows(library, ids["test"], crossings, mode="all", boundary=None)
    val_outside = float(np.mean(val_all["health"] < boundary)) if len(val_all["y"]) else 0.0
    test_outside = float(np.mean(test_all["health"] < boundary)) if len(test_all["y"]) else 0.0
    hull = {
        "train_min_capacity": boundary,
        "train_rows": int(len(train["y"])),
        "validation_rows": int(len(validation["y"])),
        "test_rows": int(len(test["y"])),
        "validation_all_rows": int(len(val_all["y"])),
        "test_all_rows": int(len(test_all["y"])),
        "validation_fraction_below_train_min": val_outside,
        "test_fraction_below_train_min": test_outside,
        "validation_scored_all_below": bool(len(validation["y"]) and np.all(validation["health"] < boundary)),
        "test_scored_all_below": bool(len(test["y"]) and np.all(test["health"] < boundary)),
    }
    print("HULL", json.dumps(hull, indent=2), flush=True)

    scored_outside = (
        hull["validation_scored_all_below"]
        and hull["test_scored_all_below"]
        and len(train["y"]) > 0
        and len(validation["y"]) > 0
        and len(test["y"]) > 0
    )
    if not scored_outside:
        write({
            "status": "infeasible",
            "reason": "a scored outer split is empty or not strictly below the actual train capacity boundary",
            "model": "PP-X v1.0 core + validation-selected residual executor",
            "protocol": "NASA_PCOE_SECOND_COHORT_V1",
            "file_sha256": manifest,
            "eligibility": audits,
            "eligible_ids": eligible,
            "split_ids": ids,
            "hull": hull,
        })
        return

    affine = select_affine_initialization(train, validation)
    pp_search = []
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
            chosen_loss = losses[decision.executor]
            pp_search[-1]["policy_executor"] = decision.executor
            pp_search[-1]["policy_reason"] = decision.reason
            # keep a compact pointer on the last row of this width/lr pair
            pp_search.append({
                "width": width, "learning_rate": lr,
                "executor": decision.executor, "validation_mse": chosen_loss,
                "policy": True, "reason": decision.reason, "extra": extras[decision.executor],
            })

    policy_rows = [row for row in pp_search if row.get("policy")]
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
        "dataset": "NASA PCoE second battery cohort (B0029–B0056 locked list)",
        "protocol": "NASA_PCOE_SECOND_COHORT_V1",
        "file_sha256": manifest,
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
            "Model-level external cohort only. Test has fewer than ten units, "
            "so this does not establish a per-unit routing gate."
        ),
    })


if __name__ == "__main__":
    main()
