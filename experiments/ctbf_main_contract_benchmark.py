#!/usr/bin/env python3
"""Thirty-candidate CTBF promotion audit on compatible PP-X main settings."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from distance_uncertainty_pp import prepare_battery
from extrapolation_competitors_all import datasets as generic_datasets
from final_modular_pp_evidence import (
    bh_adjust,
    bootstrap_ci,
    build_datasets,
    exact_sign_flip,
)
from full_equal_candidate_budget import split_hashes
from pp_extrapolation import regression_metrics
from pp_extrapolation.ctbf import fit_ctbf, predict_ctbf
from run_affine_tail_external_nasa_health_v2 import prepare_folds

OUT = ROOT / "results" / "ctbf_main_contract_benchmark"
SEARCH_SEED = 42
FINAL_SEEDS = (42, 43, 44, 45, 46)
CONFIGS = tuple(
    {"width": width, "learning_rate": learning_rate, "weight_decay": weight_decay}
    for width in (16, 32, 64)
    for learning_rate in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
    for weight_decay in (0.1, 2.0)
)
STRONGEST = {
    "sunwoda": "linear_tail_rbf",
    "rwth": "linear_tail_rbf",
    "virkler": "ft_transformer",
    "nasa": "engression",
}
PAPER_NAMES = {
    "sunwoda": "SUNWODA",
    "rwth": "RWTH",
    "virkler": "Virkler",
    "nasa": "NASA",
}


def transformed_copy(
    rows: dict,
    *,
    boundary: float,
    upper_boundary: bool,
    rate_indices: tuple[int, ...],
    mean_indices: tuple[int, ...] = (),
) -> dict:
    x = np.asarray(rows["x"], dtype=np.float64).copy()
    if upper_boundary:
        x[:, 0] = boundary - x[:, 0]
        for index in rate_indices:
            x[:, index] = -x[:, index]
        for index in mean_indices:
            x[:, index] = boundary - x[:, index]
    else:
        x[:, 0] = x[:, 0] - boundary
        for index in rate_indices:
            # Existing battery adapters store positive degradation magnitude.
            x[:, index] = -x[:, index]
        for index in mean_indices:
            x[:, index] = x[:, index] - boundary
    if np.any(x[:, 0] < -1e-6):
        raise ValueError("contract coordinate crossed below declared boundary")
    x[:, 0] = np.maximum(x[:, 0], 0.0)
    return {
        "x": x.astype(np.float32),
        "y": np.asarray(rows["y"], dtype=np.float32),
        "groups": np.asarray(rows["groups"]),
    }


def prepare_main_settings():
    generic = generic_datasets()
    settings = {}
    for name, boundary in (("sunwoda", 880.0), ("rwth", 0.8)):
        parts = generic[name]
        settings[name] = {
            "parts": tuple(
                transformed_copy(
                    rows,
                    boundary=boundary,
                    upper_boundary=False,
                    rate_indices=(1, 3),
                    mean_indices=(2,),
                )
                for rows in parts
            ),
            "rate_indices": (1, 3),
            "rate_aggregation": "median",
            "velocity_supervision_weight": 0.05,
            "boundary_definition": str(boundary),
        }
    settings["virkler"] = {
        "parts": tuple(
            transformed_copy(
                rows,
                boundary=49.8,
                upper_boundary=True,
                rate_indices=(2, 3),
            )
            for rows in generic["virkler"]
        ),
        "rate_indices": (2, 3),
        "rate_aggregation": "median",
        "velocity_supervision_weight": 0.05,
        "boundary_definition": "49.8 mm upper boundary; remaining margin",
    }
    folds, nasa_audit = prepare_folds()
    settings["nasa"] = {
        "folds": [
            {
                **fold,
                "train": {
                    key: np.asarray(value)
                    for key, value in fold["train"].items()
                    if key in ("x", "y", "groups")
                },
                "validation": {
                    key: np.asarray(value)
                    for key, value in fold["validation"].items()
                    if key in ("x", "y", "groups")
                },
                "test": {
                    key: np.asarray(value)
                    for key, value in fold["test"].items()
                    if key in ("x", "y", "groups")
                },
            }
            for fold in folds
        ],
        "rate_indices": (),
        "rate_aggregation": "first",
        "velocity_supervision_weight": 0.0,
        "boundary_definition": "health_phi=0 at 1.4 Ah",
        "nasa_audit": nasa_audit,
    }
    return settings


def fit_one(parts, config, seed, setting, *, supervision_weight):
    fitted = fit_ctbf(
        parts[0],
        parts[1],
        seed=seed,
        width=config["width"],
        boundary=0.0,
        quadrature_points=24,
        weak_rate_prior=False,
        residual_bound=1.0,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_epochs=350,
        patience=60,
        rate_indices=setting["rate_indices"],
        rate_aggregation=setting["rate_aggregation"],
        velocity_supervision_weight=supervision_weight,
    )
    return fitted


def search_and_refit(parts, setting):
    search = []
    for index, config in enumerate(CONFIGS):
        fitted = fit_one(
            parts,
            config,
            SEARCH_SEED,
            setting,
            supervision_weight=setting["velocity_supervision_weight"],
        )
        search.append(
            {
                "index": index,
                **config,
                "validation_rmse": fitted.selection["validation_rmse"],
                "selected_epoch": fitted.selection["selected_epoch"],
            }
        )
    selected = min(
        search,
        key=lambda row: (
            row["validation_rmse"],
            row["width"],
            row["learning_rate"],
            row["weight_decay"],
        ),
    )
    config = {
        key: selected[key]
        for key in ("width", "learning_rate", "weight_decay")
    }
    predictions = {"primary": [], "no_velocity_loss": []}
    epochs = {"primary": [], "no_velocity_loss": []}
    for seed in FINAL_SEEDS:
        for arm, weight in (
            ("primary", setting["velocity_supervision_weight"]),
            ("no_velocity_loss", 0.0),
        ):
            fitted = fit_one(
                parts, config, seed, setting, supervision_weight=weight
            )
            predictions[arm].append(predict_ctbf(fitted, parts[2]["x"]))
            epochs[arm].append(int(fitted.selection["selected_epoch"]))
    return (
        selected,
        search,
        {key: np.asarray(value) for key, value in predictions.items()},
        epochs,
    )


def load_equal_budget(setting, model):
    if setting != "nasa":
        z = np.load(
            ROOT
            / f"results/full_equal_candidate_budget_v1/{setting}/main/"
            f"{model}/predictions.npz",
            allow_pickle=True,
        )
        return z["y"], z["groups"].astype(str), z["prediction"]
    ys, groups, predictions = [], [], []
    for fold in range(4):
        z = np.load(
            ROOT
            / f"results/full_equal_candidate_budget_v1/nasa/fold_{fold}/"
            f"{model}/predictions.npz",
            allow_pickle=True,
        )
        ys.append(z["y"])
        groups.append(z["groups"].astype(str))
        predictions.append(z["prediction"])
    return (
        np.concatenate(ys),
        np.concatenate(groups),
        np.concatenate(predictions, axis=1),
    )


def reference_predictions(setting):
    frozen = build_datasets()[PAPER_NAMES[setting]]
    y, groups, ppx = frozen[:3]
    by, bg, strongest = load_equal_budget(setting, STRONGEST[setting])
    dy, dg, direct = load_equal_budget(setting, "plain_mlp")
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)
    if not (
        np.allclose(y, by)
        and np.allclose(y, dy)
        and np.array_equal(groups, bg)
        and np.array_equal(groups, dg)
    ):
        raise RuntimeError(f"{setting}: reference row mismatch")
    return y, groups, np.asarray(ppx), strongest, direct


def unit_effects(y, baseline, candidate, groups):
    effects = []
    ratios = []
    rows = []
    for unit in np.unique(groups):
        mask = groups == unit
        baseline_rmse = math.sqrt(float(np.mean((baseline[mask] - y[mask]) ** 2)))
        candidate_rmse = math.sqrt(
            float(np.mean((candidate[mask] - y[mask]) ** 2))
        )
        effect = math.log(max(baseline_rmse, 1e-12) / max(candidate_rmse, 1e-12))
        effects.append(effect)
        ratios.append(candidate_rmse / max(baseline_rmse, 1e-12))
        rows.append(
            {
                "unit": str(unit),
                "baseline_rmse": baseline_rmse,
                "candidate_rmse": candidate_rmse,
                "log_rmse_improvement": effect,
            }
        )
    values = np.asarray(effects)
    return {
        "mean_log_rmse_improvement": float(values.mean()),
        "ci95": bootstrap_ci(values, np.random.default_rng(20260912), reps=20000),
        "sign_flip_p": exact_sign_flip(values),
        "wins": int(np.sum(values > 0)),
        "total": int(len(values)),
        "win_fraction": float(np.mean(values > 0)),
        "worst_rmse_ratio": float(np.max(ratios)),
        "rows": rows,
    }


def evaluate_single(setting_name, setting):
    parts = setting["parts"]
    selected, search, ctbf, epochs = search_and_refit(parts, setting)
    y = np.asarray(parts[2]["y"])
    groups = np.asarray(parts[2]["groups"]).astype(str)
    ref_y, ref_groups, ppx, strongest, direct = reference_predictions(setting_name)
    if not np.allclose(y, ref_y) or not np.array_equal(groups, ref_groups):
        raise RuntimeError(f"{setting_name}: transformed rows do not match references")
    return summarize(
        setting_name,
        y,
        groups,
        ctbf,
        ppx,
        strongest,
        direct,
        selected,
        search,
        epochs,
        setting,
        split_hashes(parts),
    )


def evaluate_nasa(setting):
    fold_rows, arrays = [], {"primary": [], "no_velocity_loss": []}
    ys, groups = [], []
    for fold in setting["folds"]:
        parts = (fold["train"], fold["validation"], fold["test"])
        selected, search, predictions, epochs = search_and_refit(parts, setting)
        fold_rows.append(
            {
                "test_cell": fold["test_cell"],
                "selected": selected,
                "search": search,
                "epochs": epochs,
                "split_hashes": split_hashes(parts),
            }
        )
        ys.append(parts[2]["y"])
        groups.append(parts[2]["groups"].astype(str))
        for arm in arrays:
            arrays[arm].append(predictions[arm])
    y = np.concatenate(ys)
    group = np.concatenate(groups)
    ctbf = {
        arm: np.concatenate(values, axis=1) for arm, values in arrays.items()
    }
    ref_y, ref_groups, ppx, strongest, direct = reference_predictions("nasa")
    if not np.allclose(y, ref_y) or not np.array_equal(group, ref_groups):
        raise RuntimeError("nasa: fold rows do not match references")
    return summarize(
        "nasa",
        y,
        group,
        ctbf,
        ppx,
        strongest,
        direct,
        {"by_fold": [row["selected"] for row in fold_rows]},
        {"by_fold": [row["search"] for row in fold_rows]},
        {"by_fold": [row["epochs"] for row in fold_rows]},
        setting,
        {"by_fold": [row["split_hashes"] for row in fold_rows]},
    )


def summarize(
    name,
    y,
    groups,
    ctbf,
    ppx,
    strongest,
    direct,
    selected,
    search,
    epochs,
    setting,
    hashes,
):
    predictions = {
        "ctbf": ctbf["primary"].mean(axis=0),
        "ctbf_no_velocity_loss": ctbf["no_velocity_loss"].mean(axis=0),
        "ppx": np.asarray(ppx).mean(axis=0),
        "strongest_30c": np.asarray(strongest).mean(axis=0),
        "direct_mlp_30c": np.asarray(direct).mean(axis=0),
    }
    metrics = {
        key: regression_metrics(y, value, groups)
        for key, value in predictions.items()
    }
    probe = np.zeros((1, 1), dtype=np.float32)
    boundary_exact_by_construction = True
    comparisons = {
        "ctbf_vs_ppx": unit_effects(
            y, predictions["ppx"], predictions["ctbf"], groups
        ),
        "ctbf_vs_strongest": unit_effects(
            y, predictions["strongest_30c"], predictions["ctbf"], groups
        ),
        "ctbf_vs_direct": unit_effects(
            y, predictions["direct_mlp_30c"], predictions["ctbf"], groups
        ),
        "ctbf_vs_no_velocity_loss": unit_effects(
            y,
            predictions["ctbf_no_velocity_loss"],
            predictions["ctbf"],
            groups,
        ),
    }
    return {
        "setting": name,
        "boundary_definition": setting["boundary_definition"],
        "rate_indices": list(setting["rate_indices"]),
        "velocity_supervision_weight": setting["velocity_supervision_weight"],
        "selected": selected,
        "search": search,
        "epochs": epochs,
        "split_hashes": hashes,
        "metrics": metrics,
        "comparisons": comparisons,
        "boundary_exact_by_construction": boundary_exact_by_construction,
    }, {
        "y": y,
        "groups": groups,
        **predictions,
        "ctbf_seeds": ctbf["primary"],
        "ctbf_no_velocity_loss_seeds": ctbf["no_velocity_loss"],
    }


def main():
    torch.set_num_threads(2)
    if len(CONFIGS) != 30:
        raise RuntimeError("CTBF search grid must contain exactly 30 candidates")
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen CTBF main benchmark")
    settings = prepare_main_settings()
    results = {}
    for name in ("sunwoda", "rwth", "virkler", "nasa"):
        if name == "nasa":
            result, arrays = evaluate_nasa(settings[name])
        else:
            result, arrays = evaluate_single(name, settings[name])
        results[name] = result
        np.savez_compressed(OUT / f"{name}_predictions.npz", **arrays)
        print(
            name,
            {
                model: round(values["pooled"]["r2"], 4)
                for model, values in result["metrics"].items()
            },
            flush=True,
        )
        # Restartable without changing any completed result.
        (OUT / "results.partial.json").write_text(
            json.dumps(
                {
                    "status": "incomplete",
                    "protocol": "CTBF_MAIN_CONTRACT_BENCHMARK_PROTOCOL",
                    "datasets": results,
                },
                indent=2,
            )
            + "\n"
        )

    p_values = [
        results[name]["comparisons"]["ctbf_vs_ppx"]["sign_flip_p"]
        for name in results
    ]
    q_values = bh_adjust(p_values)
    for name, value in zip(results, q_values):
        results[name]["comparisons"]["ctbf_vs_ppx"]["bh_q"] = value
    ppx_wins = sum(
        results[name]["metrics"]["ctbf"]["pooled"]["rmse"]
        < results[name]["metrics"]["ppx"]["pooled"]["rmse"]
        for name in results
    )
    relative_harms = [
        results[name]["metrics"]["ctbf"]["pooled"]["rmse"]
        / results[name]["metrics"]["ppx"]["pooled"]["rmse"]
        - 1.0
        for name in results
    ]
    positive_unit_ci = sum(
        results[name]["comparisons"]["ctbf_vs_ppx"]["ci95"][0] > 0
        for name in results
    )
    effects = [
        results[name]["comparisons"]["ctbf_vs_ppx"][
            "mean_log_rmse_improvement"
        ]
        for name in results
    ]
    extreme = min(ppx_wins, 4 - ppx_wins)
    dataset_sign_p = min(
        1.0,
        2 * sum(math.comb(4, k) for k in range(0, extreme + 1)) / 2**4,
    )
    promotion = bool(
        ppx_wins >= 3
        and np.mean(effects) > 0
        and max(relative_harms) <= 0.10
        and positive_unit_ci >= 2
        and all(
            row["boundary_exact_by_construction"] for row in results.values()
        )
        and any(
            row["comparisons"]["ctbf_vs_no_velocity_loss"]["ci95"][0] <= 0
            for row in results.values()
        )
    )
    payload = {
        "status": "complete frozen CTBF main contract promotion audit",
        "protocol": "CTBF_MAIN_CONTRACT_BENCHMARK_PROTOCOL",
        "candidate_count": len(CONFIGS),
        "search_seed": SEARCH_SEED,
        "final_seeds": FINAL_SEEDS,
        "excluded": {
            "hust": "terminal-slope proxy target",
            "mich": "life-label target without common health boundary",
            "matr": "end-of-record target without declared boundary",
            "matr_batch2": "end-of-record target without declared boundary",
            "ncmapss": "no scalar ordered state or declared boundary",
        },
        "datasets": results,
        "aggregate": {
            "ctbf_wins_vs_ppx": int(ppx_wins),
            "datasets": 4,
            "dataset_sign_test_two_sided": float(dataset_sign_p),
            "mean_dataset_log_rmse_improvement": float(np.mean(effects)),
            "maximum_pooled_rmse_harm": float(max(relative_harms)),
            "datasets_with_positive_unit_ci": int(positive_unit_ci),
            "promote_over_ppx": promotion,
        },
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "results.partial.json").unlink(missing_ok=True)
    print("aggregate", payload["aggregate"], flush=True)


if __name__ == "__main__":
    main()
