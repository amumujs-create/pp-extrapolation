#!/usr/bin/env python3
"""Frozen retrospective CTBF challenger on Stanford and ISU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import regression_metrics
from pp_extrapolation.ctbf import (
    fit_ctbf,
    predict_ctbf,
    predict_rate_quotient,
)
from ppx_v11_complete_structure_ablation import (
    prepare_isu,
    prepare_stanford,
)

OUT = ROOT / "results" / "ctbf_stanford_isu_development"
SEEDS = (42, 43, 44)
WIDTHS = (16, 32)
BOUNDS = (0.5, 1.0, 2.0)
BOUNDARY = 0.8


def metrics(y, prediction, groups):
    return regression_metrics(y, prediction, groups)


def bootstrap(y, baseline, candidate, groups, replicates=20000):
    labels = np.unique(groups)
    positions = {label: np.flatnonzero(groups == label) for label in labels}
    rng = np.random.default_rng(20260912)
    gain = np.empty(replicates)
    for draw in range(replicates):
        sampled = rng.choice(labels, len(labels), replace=True)
        index = np.concatenate([positions[label] for label in sampled])
        gain[draw] = (
            np.sqrt(np.mean((baseline[index] - y[index]) ** 2))
            - np.sqrt(np.mean((candidate[index] - y[index]) ** 2))
        )
    return {
        "mean_rmse_improvement": float(gain.mean()),
        "ci95": [float(value) for value in np.quantile(gain, (0.025, 0.975))],
        "probability_positive": float(np.mean(gain > 0)),
    }


def unit_wins(y, baseline, candidate, groups):
    labels = np.unique(groups)
    wins = 0
    rows = []
    for label in labels:
        mask = groups == label
        baseline_rmse = float(np.sqrt(np.mean((baseline[mask] - y[mask]) ** 2)))
        candidate_rmse = float(
            np.sqrt(np.mean((candidate[mask] - y[mask]) ** 2))
        )
        wins += int(candidate_rmse < baseline_rmse)
        rows.append(
            {
                "unit": str(label),
                "baseline_rmse": baseline_rmse,
                "candidate_rmse": candidate_rmse,
                "candidate_wins": bool(candidate_rmse < baseline_rmse),
            }
        )
    return {
        "wins": int(wins),
        "total": int(len(labels)),
        "fraction": float(wins / len(labels)),
        "rows": rows,
    }


def fit_direct(train, validation, test):
    candidates = []
    predictions = {}
    for width in WIDTHS:
        key = f"direct_w{width}"
        validation_seeds, test_seeds, epochs = [], [], []
        for seed in SEEDS:
            fitted = fit_plain(
                train,
                validation,
                seed=seed,
                width=width,
                learning_rate=1e-3,
                weight_decay=0.1,
                max_epochs=350,
                patience=60,
            )
            validation_seeds.append(predict_plain(fitted, validation["x"]))
            test_seeds.append(predict_plain(fitted, test["x"]))
            epochs.append(int(fitted["selected_epoch"]))
        validation_prediction = np.mean(validation_seeds, axis=0)
        test_prediction = np.mean(test_seeds, axis=0)
        score = float(
            np.sqrt(np.mean((validation_prediction - validation["y"]) ** 2))
        )
        candidates.append(
            {"key": key, "width": width, "validation_rmse": score, "epochs": epochs}
        )
        predictions[key] = (validation_prediction, test_prediction)
    selected = min(candidates, key=lambda row: (row["validation_rmse"], row["width"]))
    return selected, predictions[selected["key"]], candidates


def fit_flow(train, validation, test, *, weak_rate_prior):
    candidates = []
    predictions = {}
    bounds = BOUNDS if weak_rate_prior else (1.0,)
    for width in WIDTHS:
        for residual_bound in bounds:
            name = "weak_rate" if weak_rate_prior else "direct_velocity"
            key = f"{name}_w{width}_b{residual_bound}"
            validation_seeds, test_seeds, selections = [], [], []
            for seed in SEEDS:
                fitted = fit_ctbf(
                    train,
                    validation,
                    seed=seed,
                    width=width,
                    boundary=BOUNDARY,
                    quadrature_points=24,
                    weak_rate_prior=weak_rate_prior,
                    residual_bound=residual_bound,
                )
                validation_seeds.append(predict_ctbf(fitted, validation["x"]))
                test_seeds.append(predict_ctbf(fitted, test["x"]))
                selections.append(fitted.selection)
            validation_prediction = np.mean(validation_seeds, axis=0)
            test_prediction = np.mean(test_seeds, axis=0)
            score = float(
                np.sqrt(np.mean((validation_prediction - validation["y"]) ** 2))
            )
            candidates.append(
                {
                    "key": key,
                    "width": width,
                    "residual_bound": residual_bound,
                    "validation_rmse": score,
                    "selections": selections,
                }
            )
            predictions[key] = (validation_prediction, test_prediction)
    selected = min(
        candidates,
        key=lambda row: (
            row["validation_rmse"],
            row["width"],
            row["residual_bound"],
        ),
    )
    return selected, predictions[selected["key"]], candidates


def evaluate(name, prepared):
    train, validation, test, ids, hull = prepared
    direct_selected, direct_predictions, direct_grid = fit_direct(
        train, validation, test
    )
    velocity_selected, velocity_predictions, velocity_grid = fit_flow(
        train, validation, test, weak_rate_prior=False
    )
    weak_selected, weak_predictions, weak_grid = fit_flow(
        train, validation, test, weak_rate_prior=True
    )
    rate_floor = float(
        np.quantile((-train["x"][:, 1])[-train["x"][:, 1] > 1e-8], 0.10)
    )
    rate_predictions = (
        predict_rate_quotient(
            validation["x"], boundary=BOUNDARY, rate_floor=rate_floor
        ),
        predict_rate_quotient(test["x"], boundary=BOUNDARY, rate_floor=rate_floor),
    )
    arms = {
        "rate_quotient": rate_predictions,
        "direct_mlp": direct_predictions,
        "direct_velocity_ctbf": velocity_predictions,
        "weak_rate_ctbf": weak_predictions,
    }
    result_arms = {}
    arrays = {}
    for arm, (validation_prediction, test_prediction) in arms.items():
        result_arms[arm] = {
            "validation": metrics(
                validation["y"], validation_prediction, validation["groups"]
            ),
            "test": metrics(test["y"], test_prediction, test["groups"]),
        }
        arrays[f"{name}_{arm}_validation"] = validation_prediction
        arrays[f"{name}_{arm}_test"] = test_prediction
    arrays[f"{name}_validation_y"] = validation["y"]
    arrays[f"{name}_test_y"] = test["y"]
    arrays[f"{name}_validation_groups"] = validation["groups"]
    arrays[f"{name}_test_groups"] = test["groups"]

    direct_test = direct_predictions[1]
    weak_test = weak_predictions[1]
    velocity_test = velocity_predictions[1]
    boundary_probe = np.asarray(test["x"][: min(32, len(test["x"]))]).copy()
    boundary_probe[:, 0] = BOUNDARY
    boundary_fits = []
    # Refit only the selected primary configuration once for an invariant probe.
    probe_fit = fit_ctbf(
        train,
        validation,
        seed=SEEDS[0],
        width=weak_selected["width"],
        boundary=BOUNDARY,
        weak_rate_prior=True,
        residual_bound=weak_selected["residual_bound"],
    )
    boundary_fits.extend(predict_ctbf(probe_fit, boundary_probe).tolist())

    comparison = {
        "weak_vs_direct_mlp_bootstrap": bootstrap(
            test["y"], direct_test, weak_test, test["groups"]
        ),
        "weak_vs_direct_mlp_unit_wins": unit_wins(
            test["y"], direct_test, weak_test, test["groups"]
        ),
        "weak_vs_direct_velocity_bootstrap": bootstrap(
            test["y"], velocity_test, weak_test, test["groups"]
        ),
        "weak_vs_direct_velocity_unit_wins": unit_wins(
            test["y"], velocity_test, weak_test, test["groups"]
        ),
        "boundary_max_abs_prediction": float(np.max(np.abs(boundary_fits))),
    }
    return {
        "cohort": name,
        "split_ids": ids,
        "hull": hull,
        "boundary": BOUNDARY,
        "selected": {
            "direct_mlp": direct_selected,
            "direct_velocity_ctbf": velocity_selected,
            "weak_rate_ctbf": weak_selected,
        },
        "grid": {
            "direct_mlp": direct_grid,
            "direct_velocity_ctbf": velocity_grid,
            "weak_rate_ctbf": weak_grid,
        },
        "arms": result_arms,
        "comparisons": comparison,
    }, arrays


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen CTBF development result")
    results = {}
    arrays = {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        result, cohort_arrays = evaluate(name, prepared)
        results[name] = result
        arrays.update(cohort_arrays)
        print(
            name,
            json.dumps(
                {
                    arm: value["test"]["pooled"]["rmse"]
                    for arm, value in result["arms"].items()
                }
            ),
            flush=True,
        )
    payload = {
        "status": "retrospective frozen CTBF model-concept challenger",
        "protocol": "CTBF_STANFORD_ISU_DEVELOPMENT_PROTOCOL",
        "seeds": SEEDS,
        "widths": WIDTHS,
        "residual_bounds": BOUNDS,
        "cohorts": results,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
