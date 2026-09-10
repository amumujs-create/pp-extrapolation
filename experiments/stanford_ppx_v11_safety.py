#!/usr/bin/env python3
"""Development replay of conservative PP-X safety continuation on Stanford."""
from __future__ import annotations

import json
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
    select_affine_initialization,
    select_prior_trust,
)
from stanford_ppx_v1 import eligibility, load_series, rows, split_ids

OUT = ROOT / "results" / "stanford_ppx_v11_safety_development"
SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
CONFIGS = tuple(
    {"width": width, "learning_rate": lr, "weight_decay": 2.0}
    for width in (16, 32)
    for lr in (5e-4, 1e-3)
)
TRUSTS = (0.02, 0.05, 0.1, 0.2, 0.4)


def prepare():
    library = load_series()
    audits = [eligibility(cell, library[cell]) for cell in sorted(library)]
    eligible = [row["cell"] for row in audits if row["eligible"]]
    crossings = {row["cell"]: int(row["crossing_index"]) for row in audits if row["eligible"]}
    ids = split_ids(eligible)
    all_train = rows(library, ids["train"], crossings, mode="all", boundary=None)
    cutoff = float(np.quantile(all_train["health"], 0.25))
    train = rows(
        library, ids["train"], crossings, mode="train",
        boundary=cutoff, train_strict=True,
    )
    boundary = float(np.min(train["health"]))
    validation = rows(library, ids["validation"], crossings, mode="tail", boundary=boundary)
    test = rows(library, ids["test"], crossings, mode="tail", boundary=boundary)
    return train, validation, test, ids, {"q25": cutoff, "boundary": boundary}


def fit_route(train, validation, test, config, trust, seeds):
    affine = select_affine_initialization(train, validation)
    val, tst, epochs = [], [], []
    for seed in seeds:
        fitted = fit_pp(
            train, validation, seed=seed, affine_selection=affine,
            max_epochs=300, patience=50, direct_residual_mixture=True,
            fixed_affine_trust=float(trust), residual_seed_replay=True,
            residual_zero_init=False, **config,
        )
        val.append(predict(fitted, validation["x"]))
        tst.append(predict(fitted, test["x"]))
        epochs.append(int(fitted.selection["selected_epoch"]))
    return np.asarray(val), np.asarray(tst), epochs


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError("refusing to overwrite completed development replay")
    train, validation, test, ids, hull = prepare()

    fallback_search = []
    fallback_runs = {}
    for config in CONFIGS:
        key = (config["width"], config["learning_rate"])
        val, tst, epochs = fit_route(
            train, validation, test, config, 0.0, SELECTION_SEEDS,
        )
        rmse = float(np.sqrt(np.mean((val.mean(0) - validation["y"]) ** 2)))
        fallback_search.append({**config, "validation_rmse": rmse, "epochs": epochs})
        fallback_runs[key] = (val, tst)
    fallback_row = min(fallback_search, key=lambda row: row["validation_rmse"])
    fallback_config = {key: fallback_row[key] for key in
                       ("width", "learning_rate", "weight_decay")}
    fallback_key = (fallback_config["width"], fallback_config["learning_rate"])
    fallback_val = fallback_runs[fallback_key][0].mean(0)

    candidate_rows, candidate_val = [], []
    for config in CONFIGS:
        for trust in TRUSTS:
            val, _, epochs = fit_route(
                train, validation, test, config, trust, SELECTION_SEEDS,
            )
            prediction = val.mean(0)
            candidate_val.append(prediction)
            candidate_rows.append({
                **config, "trust": trust,
                "validation_rmse": float(np.sqrt(np.mean(
                    (prediction - validation["y"]) ** 2
                ))),
                "epochs": epochs,
            })
    decision = select_prior_trust(
        validation["y"], validation["groups"], fallback_val,
        np.asarray(candidate_val), min_relative_gain=0.02,
        bootstrap_replicates=20000,
    )
    selected = (
        candidate_rows[decision.candidate_index]
        if decision.accepted else {**fallback_config, "trust": 0.0}
    )
    selected_config = {key: selected[key] for key in
                       ("width", "learning_rate", "weight_decay")}

    val, test_predictions, epochs = fit_route(
        train, validation, test, selected_config, selected["trust"], FINAL_SEEDS,
    )
    _, fallback_predictions, fallback_epochs = fit_route(
        train, validation, test, fallback_config, 0.0, FINAL_SEEDS,
    )
    # Audit the exact trust-zero equivalence against standalone matched MLP.
    plain = []
    for seed in FINAL_SEEDS:
        fitted = fit_plain(
            train, validation, seed=seed, max_epochs=300, patience=50,
            **fallback_config,
        )
        plain.append(predict_plain(fitted, test["x"]))
    replay_error = float(np.max(np.abs(np.asarray(plain) - fallback_predictions)))
    if replay_error > 1e-5:
        raise RuntimeError(f"trust-zero replay mismatch: {replay_error}")

    payload = {
        "status": "post-Stanford development; requires a new untouched cohort",
        "model": "PP-X v1.1 conservative safety continuation",
        "selection_rule": (
            "prior requires >=2% validation RMSE gain and positive 95% "
            "physical-unit bootstrap lower bound; otherwise exact matched MLP"
        ),
        "split_ids": ids,
        "hull": hull,
        "fallback_search": fallback_search,
        "candidate_search": candidate_rows,
        "decision": asdict(decision),
        "selected": selected,
        "selected_epochs": epochs,
        "fallback_epochs": fallback_epochs,
        "trust_zero_max_abs_replay_error": replay_error,
        "selected_validation": regression_metrics(
            validation["y"], val.mean(0), validation["groups"],
        ),
        "selected_test": regression_metrics(
            test["y"], test_predictions.mean(0), test["groups"],
        ),
        "matched_mlp_test": regression_metrics(
            test["y"], fallback_predictions.mean(0), test["groups"],
        ),
    }
    np.savez_compressed(
        OUT / "predictions.npz", y=test["y"], groups=test["groups"],
        selected=test_predictions, fallback=fallback_predictions,
    )
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "decision": payload["decision"],
        "selected": selected,
        "test": payload["selected_test"]["pooled"],
        "matched_mlp": payload["matched_mlp_test"]["pooled"],
        "replay_error": replay_error,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
