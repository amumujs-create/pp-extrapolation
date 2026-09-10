#!/usr/bin/env python3
"""Frozen PP-X v1.2 evaluation on the UConn–ISU–ILCC 1.2 Ah cohort."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import (
    fit_pp,
    predict,
    regression_metrics,
    robust_generalization_policy_config,
    select_affine_initialization,
    select_prior_trust,
)

DATA = ROOT / "data/uconn_ilcc_ppx_v12/processed_data_slowpulse_0.pkl"
OUT = ROOT / "results/uconn_ilcc_ppx_v12"
SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
CONFIGS = tuple(
    {"width": width, "learning_rate": lr, "weight_decay": 2.0}
    for width in (16, 32) for lr in (5e-4, 1e-3)
)
TRUSTS = (0.02, 0.05, 0.1, 0.2, 0.4)
HISTORY = 5
MIN_ELIGIBLE = 30
MIN_OUTER_ROWS = 50
DATASET_NAME = "UConn–ISU–ILCC 1.2 Ah LFP/Gr"
PROTOCOL_NAME = "UCONN_ILCC_LOW_CAPACITY_PPX_V12_PROTOCOL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells():
    with DATA.open("rb") as handle:
        raw = pickle.load(handle)
    cells = {}
    for cell in sorted(np.unique(raw["cell_id"]).astype(int)):
        mask = np.asarray(raw["cell_id"]) == cell
        rpt = np.asarray(raw["rpt"])[mask]
        cycle = np.asarray(raw["num_cycles"], dtype=float)[mask]
        capacity = np.asarray(raw["q_dchg"], dtype=float)[mask]
        rows = []
        for value in sorted(np.unique(rpt).astype(int)):
            take = rpt == value
            finite_capacity = capacity[take][np.isfinite(capacity[take])]
            finite_cycle = cycle[take][np.isfinite(cycle[take])]
            if finite_capacity.size and finite_cycle.size:
                rows.append((
                    float(np.median(finite_cycle)),
                    float(np.median(finite_capacity)),
                ))
        rows.sort()
        # Multiple RPTs can share the same cycle count; collapse deterministically.
        unique_cycle = sorted(set(row[0] for row in rows))
        cells[int(cell)] = {
            "cycle": np.asarray(unique_cycle, dtype=float),
            "capacity": np.asarray([
                np.median([row[1] for row in rows if row[0] == value])
                for value in unique_cycle
            ], dtype=float),
        }
    return cells


def eligibility(cell, series):
    capacity = series["capacity"]
    first5 = float(np.median(capacity[:5])) if len(capacity) >= 5 else float("nan")
    threshold = 0.8 * first5
    hits = np.flatnonzero(capacity <= threshold) if np.isfinite(threshold) else []
    crossing = int(hits[0]) if len(hits) else None
    return {
        "cell": int(cell), "observations": int(len(capacity)),
        "first5_median_ah": first5, "threshold_ah": threshold,
        "crossing_index": crossing,
        "eligible": bool(len(capacity) >= 20 and crossing is not None
                         and crossing >= HISTORY),
    }


def split_ids(eligible):
    n = len(eligible)
    a, b = int(0.6 * n), int(0.8 * n)
    return {"train": eligible[:a], "validation": eligible[a:b], "test": eligible[b:]}


def make_rows(cells, ids, crossings, *, boundary=None, train=False):
    x, y, groups, health = [], [], [], []
    for cell in ids:
        cycle = cells[cell]["cycle"]
        capacity = cells[cell]["capacity"]
        scale = float(np.median(capacity[:5]))
        h = capacity / scale
        stop = crossings[cell]
        for index in range(HISTORY - 1, stop):
            value = float(h[index])
            if boundary is not None:
                if train and value <= boundary:
                    continue
                if not train and value >= boundary:
                    continue
            window = h[index - HISTORY + 1:index + 1]
            slopes = []
            for lag in (1, 3, 4):
                start = max(0, index - lag)
                delta_cycle = max(float(cycle[index] - cycle[start]), 1.0)
                slopes.append(float((h[index] - h[start]) / delta_cycle))
            x.append([
                value, *slopes, float(window.mean()), float(window.std()),
                float(cycle[index]),
            ])
            y.append(float(cycle[stop] - cycle[index]))
            groups.append(cell)
            health.append(value)
    return {
        "x": np.asarray(x, np.float32),
        "y": np.asarray(y, np.float32),
        "groups": np.asarray(groups),
        "health": np.asarray(health, float),
    }


def fit_route(train, validation, test, config, trust, seeds):
    affine = select_affine_initialization(train, validation)
    validation_predictions, test_predictions, epochs = [], [], []
    for seed in seeds:
        fitted = fit_pp(
            train, validation, seed=seed, affine_selection=affine,
            max_epochs=300, patience=50, direct_residual_mixture=True,
            fixed_affine_trust=float(trust), residual_seed_replay=True,
            residual_zero_init=False, **config,
        )
        validation_predictions.append(predict(fitted, validation["x"]))
        test_predictions.append(predict(fitted, test["x"]))
        epochs.append(int(fitted.selection["selected_epoch"]))
    return np.asarray(validation_predictions), np.asarray(test_predictions), epochs


def paired_bootstrap(y, selected, fallback, groups, replicates=20000):
    labels = np.unique(groups)
    positions = {label: np.flatnonzero(groups == label) for label in labels}
    rng = np.random.default_rng(20260910)
    differences = np.empty(replicates)
    for draw in range(replicates):
        sampled = rng.choice(labels, len(labels), replace=True)
        index = np.concatenate([positions[label] for label in sampled])
        differences[draw] = (
            np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
            - np.sqrt(np.mean((selected[index] - y[index]) ** 2))
        )
    return {
        "mean": float(differences.mean()),
        "95ci": [float(x) for x in np.quantile(differences, (0.025, 0.975))],
        "probability_positive": float(np.mean(differences > 0)),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite first UConn outcome")

    cells = load_cells()
    audit = [eligibility(cell, cells[cell]) for cell in sorted(cells)]
    eligible = [row["cell"] for row in audit if row["eligible"]]
    crossings = {row["cell"]: row["crossing_index"] for row in audit if row["eligible"]}
    if len(eligible) < MIN_ELIGIBLE:
        result_path.write_text(json.dumps({
            "status": "inconclusive",
            "reason": f"fewer than {MIN_ELIGIBLE} eligible cells",
            "sha256": sha256(DATA), "eligibility": audit,
        }, indent=2) + "\n")
        return
    ids = split_ids(eligible)
    if len(ids["validation"]) < 6 or len(ids["test"]) < 6:
        raise RuntimeError("frozen split has insufficient outer units")

    train_all = make_rows(cells, ids["train"], crossings)
    cutoff = float(np.quantile(train_all["health"], 0.25))
    train = make_rows(cells, ids["train"], crossings, boundary=cutoff, train=True)
    boundary = float(np.min(train["health"]))
    validation = make_rows(cells, ids["validation"], crossings, boundary=boundary)
    test = make_rows(cells, ids["test"], crossings, boundary=boundary)
    hull = {
        "train_health_q25": cutoff, "actual_train_min": boundary,
        "train_rows": len(train["y"]), "validation_rows": len(validation["y"]),
        "test_rows": len(test["y"]),
        "validation_outside_fraction": float(np.mean(validation["health"] < boundary))
        if len(validation["y"]) else 0.0,
        "test_outside_fraction": float(np.mean(test["health"] < boundary))
        if len(test["y"]) else 0.0,
    }
    if (len(validation["y"]) < MIN_OUTER_ROWS
            or len(test["y"]) < MIN_OUTER_ROWS):
        result_path.write_text(json.dumps({
            "status": "infeasible",
            "reason": f"outer split has fewer than {MIN_OUTER_ROWS} rows",
            "sha256": sha256(DATA), "eligibility": audit, "split_ids": ids,
            "hull": hull,
        }, indent=2) + "\n")
        return

    fallback_search, fallback_predictions = [], {}
    for config in CONFIGS:
        validation_run, _, epochs = fit_route(
            train, validation, test, config, 0.0, SELECTION_SEEDS,
        )
        prediction = validation_run.mean(0)
        key = (config["width"], config["learning_rate"])
        fallback_predictions[key] = prediction
        fallback_search.append({
            **config, "validation_rmse": float(np.sqrt(np.mean(
                (prediction - validation["y"]) ** 2
            ))), "epochs": epochs,
        })
    fallback_row = min(fallback_search, key=lambda row: row["validation_rmse"])
    fallback_config = {key: fallback_row[key] for key in
                       ("width", "learning_rate", "weight_decay")}
    fallback_key = (fallback_config["width"], fallback_config["learning_rate"])
    fallback_validation = fallback_predictions[fallback_key]

    candidates, candidate_predictions = [], []
    for config in CONFIGS:
        for trust in TRUSTS:
            validation_run, _, epochs = fit_route(
                train, validation, test, config, trust, SELECTION_SEEDS,
            )
            prediction = validation_run.mean(0)
            candidate_predictions.append(prediction)
            candidates.append({
                **config, "trust": trust,
                "validation_rmse": float(np.sqrt(np.mean(
                    (prediction - validation["y"]) ** 2
                ))), "epochs": epochs,
            })
    policy = robust_generalization_policy_config()
    decision = select_prior_trust(
        validation["y"], validation["groups"], fallback_validation,
        np.asarray(candidate_predictions), **policy,
    )
    selected = candidates[decision.candidate_index] if decision.accepted else {
        **fallback_config, "trust": 0.0,
    }
    selected_config = {key: selected[key] for key in
                       ("width", "learning_rate", "weight_decay")}
    selected_validation, selected_test, selected_epochs = fit_route(
        train, validation, test, selected_config, selected["trust"], FINAL_SEEDS,
    )
    _, fallback_test, fallback_epochs = fit_route(
        train, validation, test, fallback_config, 0.0, FINAL_SEEDS,
    )
    plain_test = []
    for seed in FINAL_SEEDS:
        fitted = fit_plain(
            train, validation, seed=seed, max_epochs=300, patience=50,
            **fallback_config,
        )
        plain_test.append(predict_plain(fitted, test["x"]))
    replay_error = float(np.max(np.abs(np.asarray(plain_test) - fallback_test)))
    if replay_error > 1e-5:
        raise RuntimeError(f"trust-zero replay mismatch: {replay_error}")

    selected_ensemble = selected_test.mean(0)
    fallback_ensemble = fallback_test.mean(0)
    selected_metrics = regression_metrics(test["y"], selected_ensemble, test["groups"])
    fallback_metrics = regression_metrics(test["y"], fallback_ensemble, test["groups"])
    validation_metrics = regression_metrics(
        validation["y"], selected_validation.mean(0), validation["groups"],
    )
    payload = {
        "status": "first frozen PP-X v1.2 external evaluation complete",
        "dataset": DATASET_NAME,
        "protocol": PROTOCOL_NAME,
        "sha256": sha256(DATA), "eligibility": audit, "eligible_ids": eligible,
        "split_ids": ids, "hull": hull, "policy": policy,
        "fallback_search": fallback_search, "candidate_search": candidates,
        "decision": asdict(decision), "selected": selected,
        "selected_epochs": selected_epochs, "fallback_epochs": fallback_epochs,
        "trust_zero_max_abs_replay_error": replay_error,
        "selected_validation": validation_metrics,
        "selected_test": selected_metrics, "matched_mlp_test": fallback_metrics,
        "paired_cell_bootstrap": paired_bootstrap(
            test["y"], selected_ensemble, fallback_ensemble, test["groups"],
        ),
        "validation_certified": bool(
            validation_metrics["pooled"]["r2"] is not None
            and validation_metrics["pooled"]["r2"] > 0
        ),
        "confirmatory_model_success": bool(
            selected_metrics["pooled"]["r2"] is not None
            and selected_metrics["pooled"]["r2"] > 0
            and selected_metrics["pooled"]["rmse"]
            <= fallback_metrics["pooled"]["rmse"] + 1e-8
        ),
        "prior_expansion_success": bool(
            decision.accepted and selected_metrics["pooled"]["r2"] > 0
        ),
    }
    np.savez_compressed(
        OUT / "predictions.npz", y=test["y"], groups=test["groups"],
        selected=selected_test, fallback=fallback_test, health=test["health"],
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "eligible": len(eligible), "split": {k: len(v) for k, v in ids.items()},
        "hull": hull, "decision": payload["decision"],
        "validation": validation_metrics["pooled"],
        "selected_test": selected_metrics["pooled"],
        "matched_mlp_test": fallback_metrics["pooled"],
        "model_success": payload["confirmatory_model_success"],
        "prior_expansion_success": payload["prior_expansion_success"],
        "replay_error": replay_error,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
