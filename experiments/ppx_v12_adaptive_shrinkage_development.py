#!/usr/bin/env python3
"""Retrospective development replay of PP-X v1.2 adaptive prior shrinkage."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from pp_extrapolation import (
    adaptive_shrinkage_policy_config,
    apply_adaptive_shrinkage,
    regression_metrics,
    select_adaptive_prior_shrinkage,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
OUT = ROOT / "results/ppx_v12_adaptive_shrinkage_development"


def rmse(y, prediction):
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def classical_fallback_audit(train, validation, test):
    models = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        models.append((
            f"ridge_{alpha}",
            make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        ))
    for leaf in (5, 15, 30):
        models.extend([
            (
                f"extra_trees_leaf_{leaf}",
                ExtraTreesRegressor(
                    n_estimators=300, min_samples_leaf=leaf,
                    random_state=42, n_jobs=-1,
                ),
            ),
            (
                f"hist_gb_leaf_{leaf}",
                HistGradientBoostingRegressor(
                    max_iter=300, l2_regularization=2.0,
                    min_samples_leaf=leaf, random_state=42,
                ),
            ),
        ])
    rows = []
    for name, model in models:
        model.fit(train["x"], train["y"])
        validation_prediction = np.clip(model.predict(validation["x"]), 0, None)
        test_prediction = np.clip(model.predict(test["x"]), 0, None)
        rows.append({
            "model": name,
            "validation_rmse": rmse(validation["y"], validation_prediction),
            "test": regression_metrics(
                test["y"], test_prediction, test["groups"],
            ),
        })
    return rows


def replay(name, prepared, source, predictions):
    train, validation, test, _, _ = prepared
    cohort = source["cohorts"][name]
    fallback_key = cohort["fallback"]
    fallback_validation = predictions[
        f"{name}_{fallback_key}_validation"
    ].mean(0)
    fallback_test = predictions[f"{name}_{fallback_key}_test"].mean(0)
    positive_rows = [row for row in cohort["grid_rows"] if row["trust"] > 0]
    candidate_validation = np.asarray([
        predictions[f"{name}_{row['key']}_validation"].mean(0)
        for row in positive_rows
    ])
    candidate_test = np.asarray([
        predictions[f"{name}_{row['key']}_test"].mean(0)
        for row in positive_rows
    ])
    decision = select_adaptive_prior_shrinkage(
        validation["y"], validation["groups"], fallback_validation,
        candidate_validation,
        np.asarray([row["trust"] for row in positive_rows]),
        **adaptive_shrinkage_policy_config(),
    )
    selected_test = apply_adaptive_shrinkage(
        fallback_test, candidate_test[decision.candidate_index],
        decision.prior_weight,
    )
    selected_validation = apply_adaptive_shrinkage(
        fallback_validation, candidate_validation[decision.candidate_index],
        decision.prior_weight,
    )
    classical = classical_fallback_audit(train, validation, test)
    return {
        "decision": asdict(decision),
        "selected_candidate": positive_rows[decision.candidate_index]["key"],
        "v12_validation": regression_metrics(
            validation["y"], selected_validation, validation["groups"],
        ),
        "v12_test": regression_metrics(
            test["y"], selected_test, test["groups"],
        ),
        "v11_hard_gate_test": regression_metrics(
            test["y"], fallback_test, test["groups"],
        ),
        "always_on_test": regression_metrics(
            test["y"], candidate_test[decision.candidate_index], test["groups"],
        ),
        "classical_fallback_audit": classical,
        "best_classical_by_validation": min(
            classical, key=lambda row: row["validation_rmse"]
        )["model"],
    }, selected_test


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite v1.2 development replay")
    source = json.loads((SOURCE / "results.json").read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        result, selected = replay(name, prepared, source, predictions)
        results[name] = result
        arrays[name] = selected
        print(name, json.dumps({
            "decision": result["decision"],
            "v12_test": result["v12_test"]["pooled"],
            "v11_test": result["v11_hard_gate_test"]["pooled"],
            "always_on_test": result["always_on_test"]["pooled"],
            "best_classical": result["best_classical_by_validation"],
        }), flush=True)
    payload = {
        "status": "post-test v1.2 development; requires untouched confirmation",
        "model": "PP-X v1.2 adaptive prior shrinkage",
        "policy": adaptive_shrinkage_policy_config(),
        "cohorts": results,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
