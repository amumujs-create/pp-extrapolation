#!/usr/bin/env python3
"""Frozen Stage A: contract normalization and multiscale CTBF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import isu_ilcc_250mah_ppx_v12 as isu
from ctbf_stanford_isu_development import bootstrap, fit_direct, unit_wins
from pp_extrapolation import regression_metrics
from pp_extrapolation.ctbf import (
    contract_normalize_features,
    fit_ctbf,
    predict_ctbf,
)
from ppx_v11_complete_structure_ablation import (
    prepare_isu,
    prepare_stanford,
)

OUT = ROOT / "results" / "ctbf_contract_normalized_stage_a"
SEEDS = (42, 43, 44)
WIDTHS = (16, 32)
BOUNDS = (0.5, 1.0, 2.0)


def subset_with_contract(
    rows: dict,
    boundaries: dict,
    *,
    rate_indices: tuple[int, ...],
    mean_indices: tuple[int, ...],
    std_indices: tuple[int, ...] = (),
) -> dict:
    row_boundaries = np.asarray(
        [boundaries[str(group)] for group in rows["groups"]], dtype=float
    )
    return {
        **rows,
        "x": contract_normalize_features(
            rows["x"],
            row_boundaries,
            rate_indices=rate_indices,
            mean_indices=mean_indices,
            std_indices=std_indices,
        ),
        "contract_boundary": row_boundaries,
    }


def prepare_contract_stanford():
    train, validation, test, ids, hull = prepare_stanford()
    boundaries = {
        str(group): 0.8
        for split in (train, validation, test)
        for group in np.unique(split["groups"])
    }
    transform = lambda rows: subset_with_contract(  # noqa: E731
        rows,
        boundaries,
        rate_indices=(1, 3),
        mean_indices=(2,),
    )
    return transform(train), transform(validation), transform(test), ids, {
        **hull,
        "contract_boundary_min": 0.8,
        "contract_boundary_max": 0.8,
    }, (1, 3)


def prepare_contract_isu():
    train, validation, test, ids, hull = prepare_isu()
    cells = isu.load_cells()
    boundaries = {
        str(cell): 0.200 / float(np.median(series["capacity"][:5]))
        for cell, series in cells.items()
    }
    transform = lambda rows: subset_with_contract(  # noqa: E731
        rows,
        boundaries,
        rate_indices=(1, 2, 3),
        mean_indices=(4,),
        std_indices=(5,),
    )
    used = [
        boundaries[str(group)]
        for split in (train, validation, test)
        for group in np.unique(split["groups"])
    ]
    return transform(train), transform(validation), transform(test), ids, {
        **hull,
        "contract_boundary_min": float(np.min(used)),
        "contract_boundary_max": float(np.max(used)),
        "contract_boundary_median": float(np.median(used)),
    }, (1, 2, 3)


def fit_flow(train, validation, test, *, rate_indices, aggregation):
    candidates = []
    predictions = {}
    weak = aggregation != "none"
    bounds = BOUNDS if weak else (1.0,)
    indices = rate_indices if aggregation == "median" else (rate_indices[0],)
    for width in WIDTHS:
        for residual_bound in bounds:
            key = f"{aggregation}_w{width}_b{residual_bound}"
            validation_seeds, test_seeds, selections = [], [], []
            for seed in SEEDS:
                fitted = fit_ctbf(
                    train,
                    validation,
                    seed=seed,
                    width=width,
                    boundary=0.0,
                    quadrature_points=24,
                    weak_rate_prior=weak,
                    residual_bound=residual_bound,
                    rate_indices=indices,
                    rate_aggregation=(
                        "median" if aggregation == "median" else "first"
                    ),
                )
                validation_seeds.append(predict_ctbf(fitted, validation["x"]))
                test_seeds.append(predict_ctbf(fitted, test["x"]))
                selections.append(
                    {
                        key: value
                        for key, value in fitted.selection.items()
                        if key != "history"
                    }
                )
            validation_prediction = np.mean(validation_seeds, axis=0)
            test_prediction = np.mean(test_seeds, axis=0)
            validation_rmse = float(
                np.sqrt(np.mean((validation_prediction - validation["y"]) ** 2))
            )
            candidates.append(
                {
                    "key": key,
                    "width": width,
                    "residual_bound": residual_bound,
                    "validation_rmse": validation_rmse,
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


def evaluate(name, prepared, stored_ppx):
    train, validation, test, ids, contract, rate_indices = prepared
    direct_selected, direct_prediction, direct_grid = fit_direct(
        train, validation, test
    )
    flow_results = {}
    for aggregation in ("none", "first", "median"):
        selected, prediction, grid = fit_flow(
            train,
            validation,
            test,
            rate_indices=rate_indices,
            aggregation=aggregation,
        )
        flow_results[aggregation] = {
            "selected": selected,
            "prediction": prediction,
            "grid": grid,
        }

    arms = {"direct_mlp": direct_prediction}
    arms.update(
        {
            f"{aggregation}_ctbf": values["prediction"]
            for aggregation, values in flow_results.items()
        }
    )
    arrays = {}
    arm_metrics = {}
    for arm, (validation_prediction, test_prediction) in arms.items():
        arm_metrics[arm] = {
            "validation": regression_metrics(
                validation["y"], validation_prediction, validation["groups"]
            ),
            "test": regression_metrics(
                test["y"], test_prediction, test["groups"]
            ),
        }
        arrays[f"{name}_{arm}_validation"] = validation_prediction
        arrays[f"{name}_{arm}_test"] = test_prediction
    arrays[f"{name}_validation_y"] = validation["y"]
    arrays[f"{name}_test_y"] = test["y"]
    arrays[f"{name}_validation_groups"] = validation["groups"]
    arrays[f"{name}_test_groups"] = test["groups"]

    direct_test = direct_prediction[1]
    lag_test = flow_results["first"]["prediction"][1]
    multi_test = flow_results["median"]["prediction"][1]
    probe = np.asarray(test["x"][: min(32, len(test["x"]))]).copy()
    probe[:, 0] = 0.0
    selected = flow_results["median"]["selected"]
    probe_fit = fit_ctbf(
        train,
        validation,
        seed=SEEDS[0],
        width=selected["width"],
        boundary=0.0,
        weak_rate_prior=True,
        residual_bound=selected["residual_bound"],
        rate_indices=rate_indices,
        rate_aggregation="median",
    )
    boundary_prediction = predict_ctbf(probe_fit, probe)
    comparisons = {
        "multi_vs_direct_bootstrap": bootstrap(
            test["y"], direct_test, multi_test, test["groups"]
        ),
        "multi_vs_direct_unit_wins": unit_wins(
            test["y"], direct_test, multi_test, test["groups"]
        ),
        "multi_vs_lag1_bootstrap": bootstrap(
            test["y"], lag_test, multi_test, test["groups"]
        ),
        "multi_vs_lag1_unit_wins": unit_wins(
            test["y"], lag_test, multi_test, test["groups"]
        ),
        "boundary_max_abs_prediction": float(
            np.max(np.abs(boundary_prediction))
        ),
    }
    return {
        "cohort": name,
        "split_ids": ids,
        "contract": contract,
        "selected": {
            "direct_mlp": direct_selected,
            **{
                f"{aggregation}_ctbf": values["selected"]
                for aggregation, values in flow_results.items()
            },
        },
        "grid": {
            "direct_mlp": direct_grid,
            **{
                f"{aggregation}_ctbf": values["grid"]
                for aggregation, values in flow_results.items()
            },
        },
        "arms": arm_metrics,
        "stored_ppx_v11": stored_ppx,
        "comparisons": comparisons,
    }, arrays


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen Stage A result")

    stored = json.loads(
        (
            ROOT
            / "results/ppx_v11_complete_structure_ablation/results.json"
        ).read_text()
    )["cohorts"]
    stored_ppx = {
        name: values["gate_2x2"]["margin_1_bootstrap_1"]["test"]
        for name, values in stored.items()
    }
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_contract_stanford()),
        ("ISU_250mAh", prepare_contract_isu()),
    ):
        result, cohort_arrays = evaluate(name, prepared, stored_ppx[name])
        results[name] = result
        arrays.update(cohort_arrays)
        print(
            name,
            json.dumps(
                {
                    arm: values["test"]["pooled"]["rmse"]
                    for arm, values in result["arms"].items()
                }
            ),
            flush=True,
        )

    stanford_rmse = results["Stanford"]["arms"]["median_ctbf"]["test"]["pooled"][
        "rmse"
    ]
    isu_rmse = results["ISU_250mAh"]["arms"]["median_ctbf"]["test"]["pooled"][
        "rmse"
    ]
    stage_a_pass = bool(stanford_rmse <= 75.0 and isu_rmse < 2.5)
    payload = {
        "status": "retrospective frozen contract-normalized CTBF Stage A",
        "protocol": "CTBF_CONTRACT_NORMALIZED_STAGE_A_PROTOCOL",
        "seeds": SEEDS,
        "widths": WIDTHS,
        "residual_bounds": BOUNDS,
        "stage_a_pass": stage_a_pass,
        "cohorts": results,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    print("stage_a_pass", stage_a_pass, flush=True)


if __name__ == "__main__":
    main()
