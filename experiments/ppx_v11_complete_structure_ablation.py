#!/usr/bin/env python3
"""Complete implemented PP-X v1.1 structure ablation on two reused cohorts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import isu_ilcc_250mah_ppx_v12 as isu
import uconn_ilcc_ppx_v12 as common
from pp_extrapolation import regression_metrics
from stanford_ppx_v11_safety import fit_route, prepare as prepare_stanford

OUT = ROOT / "results/ppx_v11_complete_structure_ablation"
CONFIGS = tuple(
    {"width": width, "learning_rate": lr, "weight_decay": 2.0}
    for width in (16, 32) for lr in (5e-4, 1e-3)
)
TRUSTS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
SEEDS = (42, 43, 44, 45, 46)


def prepare_isu():
    cells = isu.load_cells()
    audit = [isu.eligibility(cell, cells[cell]) for cell in sorted(
        cells, key=isu.numeric_id
    )]
    eligible = [row["cell"] for row in audit if row["eligible"]]
    crossing = {row["cell"]: row["crossing_index"] for row in audit
                if row["eligible"]}
    ids = isu.split_ids(eligible)
    all_train = common.make_rows(cells, ids["train"], crossing)
    cutoff = float(np.quantile(all_train["health"], 0.25))
    train = common.make_rows(
        cells, ids["train"], crossing, boundary=cutoff, train=True,
    )
    boundary = float(np.min(train["health"]))
    validation = common.make_rows(
        cells, ids["validation"], crossing, boundary=boundary,
    )
    test = common.make_rows(cells, ids["test"], crossing, boundary=boundary)
    return train, validation, test, ids, {
        "q25": cutoff, "boundary": boundary,
    }


def unit_bootstrap(y, fallback, candidate, groups, replicates=20000):
    labels = np.unique(groups)
    positions = {label: np.flatnonzero(groups == label) for label in labels}
    rng = np.random.default_rng(20260910)
    values = np.empty(replicates)
    for draw in range(replicates):
        sampled = rng.choice(labels, len(labels), replace=True)
        index = np.concatenate([positions[label] for label in sampled])
        values[draw] = (
            np.sqrt(np.mean((fallback[index] - y[index]) ** 2))
            - np.sqrt(np.mean((candidate[index] - y[index]) ** 2))
        )
    return {
        "mean_rmse_improvement": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(values, (0.025, 0.975))],
        "probability_positive": float(np.mean(values > 0)),
    }


def pooled_seed_metrics(y, predictions):
    return [regression_metrics(y, prediction, np.arange(len(y)))["pooled"]
            for prediction in predictions]


def evaluate(name, prepared):
    train, validation, test, ids, hull = prepared
    rows, predictions = [], {}
    for config in CONFIGS:
        for trust in TRUSTS:
            val, tst, epochs = fit_route(
                train, validation, test, config, trust, SEEDS,
            )
            key = f"w{config['width']}_lr{config['learning_rate']}_t{trust}"
            predictions[key] = {"validation": val, "test": tst}
            val_metrics = regression_metrics(
                validation["y"], val.mean(0), validation["groups"],
            )
            test_metrics = regression_metrics(
                test["y"], tst.mean(0), test["groups"],
            )
            rows.append({
                **config, "trust": trust, "key": key, "epochs": epochs,
                "validation": val_metrics, "test": test_metrics,
                "test_seed_metrics": pooled_seed_metrics(test["y"], tst),
            })

    fallback = min(
        (row for row in rows if row["trust"] == 0),
        key=lambda row: row["validation"]["pooled"]["rmse"],
    )
    fallback_val = predictions[fallback["key"]]["validation"].mean(0)
    fallback_test = predictions[fallback["key"]]["test"].mean(0)
    positive = min(
        (row for row in rows if row["trust"] > 0),
        key=lambda row: row["validation"]["pooled"]["rmse"],
    )
    candidate_val = predictions[positive["key"]]["validation"].mean(0)
    candidate_test = predictions[positive["key"]]["test"].mean(0)
    fallback_rmse = fallback["validation"]["pooled"]["rmse"]
    candidate_rmse = positive["validation"]["pooled"]["rmse"]
    relative_gain = float((fallback_rmse - candidate_rmse) / fallback_rmse)
    bootstrap = unit_bootstrap(
        validation["y"], fallback_val, candidate_val, validation["groups"],
    )

    unit_labels = np.unique(validation["groups"])
    unit_wins = 0
    for label in unit_labels:
        mask = validation["groups"] == label
        fallback_error = np.sqrt(np.mean(
            (fallback_val[mask] - validation["y"][mask]) ** 2
        ))
        candidate_error = np.sqrt(np.mean(
            (candidate_val[mask] - validation["y"][mask]) ** 2
        ))
        unit_wins += int(candidate_error < fallback_error)

    gate_arms = {}
    for margin_on in (False, True):
        for bootstrap_on in (False, True):
            accepted = (
                (not margin_on or relative_gain >= 0.02)
                and (not bootstrap_on or bootstrap["ci95"][0] > 0)
            )
            selected_prediction = candidate_test if accepted else fallback_test
            gate_arms[f"margin_{int(margin_on)}_bootstrap_{int(bootstrap_on)}"] = {
                "accepted": accepted,
                "selected": positive["key"] if accepted else fallback["key"],
                "test": regression_metrics(
                    test["y"], selected_prediction, test["groups"],
                ),
            }

    by_trust = []
    for trust in TRUSTS:
        best = min(
            (row for row in rows if row["trust"] == trust),
            key=lambda row: row["validation"]["pooled"]["rmse"],
        )
        by_trust.append({
            "trust": trust, "selected_key": best["key"],
            "validation": best["validation"], "test": best["test"],
        })

    return {
        "cohort": name, "split_ids": ids, "hull": hull,
        "grid_rows": rows, "trust_dose": by_trust,
        "fallback": fallback["key"], "always_on": positive["key"],
        "relative_validation_rmse_gain": relative_gain,
        "unit_wins": int(unit_wins), "unit_total": int(len(unit_labels)),
        "unit_win_fraction": float(unit_wins / len(unit_labels)),
        "bootstrap": bootstrap, "gate_2x2": gate_arms,
        "full_gate_equals_fallback_max_abs": float(np.max(np.abs(
            (candidate_test if (
                relative_gain >= 0.02 and bootstrap["ci95"][0] > 0
            ) else fallback_test) - fallback_test
        ))),
    }, {
        f"{name}_{key}_{split}": value[split]
        for key, value in predictions.items()
        for split in ("validation", "test")
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite complete v1.1 ablation")
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        result, cohort_arrays = evaluate(name, prepared)
        results[name] = result
        arrays.update(cohort_arrays)
        print(name, json.dumps({
            "always_on": result["always_on"],
            "gain": result["relative_validation_rmse_gain"],
            "bootstrap": result["bootstrap"],
            "gate": result["gate_2x2"],
        }), flush=True)
    payload = {
        "status": "retrospective complete implemented-v1.1 structural ablation",
        "protocol": "PPX_V11_COMPLETE_STRUCTURE_ABLATION_PROTOCOL",
        "configs": CONFIGS, "trusts": TRUSTS, "seeds": SEEDS,
        "cohorts": results,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
