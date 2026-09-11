#!/usr/bin/env python3
"""Retrospective test of validation-calibrated residual authority."""
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
from ppx_v11_complete_structure_ablation import prepare_isu
from pp_extrapolation import fit_pp, regression_metrics, select_affine_initialization
from pp_extrapolation.model import transform_features
from pp_extrapolation.residual_authority import (
    apply_residual_authority,
    crossfit_residual_authority,
    distance_shell_edges,
)
from pp_extrapolation.risk_budgeted_prior import (
    fit_support_scale,
    group_mse,
    support_distance,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

OUT = ROOT / "results" / "residual_authority_development"
PROTOCOL = "protocols/RESIDUAL_AUTHORITY_DEVELOPMENT_PROTOCOL.md"
CONFIGS = (
    {"width": 16, "learning_rate": 1e-3, "weight_decay": 2.0},
    {"width": 32, "learning_rate": 1e-3, "weight_decay": 2.0},
)
SEEDS = (42, 43, 44)
AUTHORITY_SETTINGS = {
    "authority_grid": (0.0, 0.25, 0.50, 0.75, 1.0),
    "epsilon": 0.02,
    "cvar_fraction": 0.20,
    "minimum_groups_per_shell": 2,
    "minimum_relative_gain": 0.0,
}


def subset(rows: dict, mask: np.ndarray) -> dict:
    return {
        key: np.asarray(value)[mask]
        for key, value in rows.items()
    }


def inner_group_split(train: dict) -> tuple[dict, dict, dict]:
    labels = np.asarray(sorted(np.unique(train["groups"]), key=str))
    inner_labels = set(labels[4::5])
    if len(inner_labels) < 2:
        inner_labels = set(labels[-2:])
    inner_mask = np.asarray([
        label in inner_labels for label in train["groups"]
    ])
    return (
        subset(train, ~inner_mask),
        subset(train, inner_mask),
        {
            "model_units": [
                str(value) for value in labels if value not in inner_labels
            ],
            "checkpoint_units": [
                str(value) for value in labels if value in inner_labels
            ],
        },
    )


def pp_components(fitted, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fitted.model.eval()
    value = torch.as_tensor(
        transform_features(x, fitted.center, fitted.scale),
        dtype=torch.float32,
    )
    with torch.no_grad():
        affine, correction = fitted.model.components(value)
        prior = affine.squeeze(1).cpu().numpy() * fitted.target_scale
        candidate = (
            affine.squeeze(1) + correction.squeeze(1)
        ).cpu().numpy() * fitted.target_scale
    return (
        np.clip(prior, 0.0, fitted.target_scale).astype(np.float64),
        np.clip(candidate, 0.0, fitted.target_scale).astype(np.float64),
    )


def positive_scale(values: np.ndarray) -> float:
    positive = np.asarray(values)[np.asarray(values) > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def fit_configuration(
    model_train: dict,
    checkpoint: dict,
    validation: dict,
    test: dict,
    config: dict,
) -> dict:
    affine = select_affine_initialization(model_train, checkpoint)
    arrays = {
        split: {"fallback": [], "prior": [], "candidate": []}
        for split in ("validation", "test")
    }
    epochs = {"fallback": [], "candidate": []}
    for seed in SEEDS:
        fallback_fit = fit_plain(
            model_train, checkpoint, seed=seed, max_epochs=300, patience=50,
            **config,
        )
        candidate_fit = fit_pp(
            model_train, checkpoint, seed=seed, affine_selection=affine,
            max_epochs=300, patience=50, residual_decay=0.3,
            residual_zero_init=False, **config,
        )
        epochs["fallback"].append(int(fallback_fit["selected_epoch"]))
        epochs["candidate"].append(
            int(candidate_fit.selection["selected_epoch"])
        )
        for split, rows in (("validation", validation), ("test", test)):
            arrays[split]["fallback"].append(
                predict_plain(fallback_fit, rows["x"])
            )
            prior, candidate = pp_components(candidate_fit, rows["x"])
            arrays[split]["prior"].append(prior)
            arrays[split]["candidate"].append(candidate)
    return {
        "config": config,
        "epochs": epochs,
        "arrays": {
            split: {
                name: np.mean(np.asarray(values), axis=0)
                for name, values in components.items()
            }
            for split, components in arrays.items()
        },
    }


def harm_ratios(y, groups, fallback, prediction, fraction=0.20):
    baseline_loss = group_mse(y, fallback, groups)
    model_loss = group_mse(y, prediction, groups)
    excess = model_loss - baseline_loss
    scale = max(float(np.mean(baseline_loss)), 1e-12)
    count = max(1, int(np.ceil(fraction * len(excess))))
    return {
        "mean_excess_ratio": float(np.mean(excess) / scale),
        "tail_excess_ratio": float(
            np.mean(np.sort(excess)[-count:]) / scale
        ),
    }


def calibrate_arms(validation, components, distance):
    edges = distance_shell_edges(distance)
    common = dict(
        y=validation["y"],
        groups=validation["groups"],
        fallback=components["fallback"],
        prior=components["prior"],
        candidate=components["candidate"],
        distance=distance,
        **AUTHORITY_SETTINGS,
    )
    return {
        "global_authority": crossfit_residual_authority(
            **common, edges=(0.0, float("inf")),
        ),
        "free_shell_authority": crossfit_residual_authority(
            **common, edges=edges, enforce_monotone=False,
        ),
        "monotone_shell_authority": crossfit_residual_authority(
            **common, edges=edges, enforce_monotone=True,
        ),
    }


def evaluate(name: str, prepared) -> tuple[dict, dict]:
    train, validation, test, ids, hull = prepared
    model_train, checkpoint, inner_ids = inner_group_split(train)
    support = fit_support_scale(model_train["x"])
    val_distance_raw = support_distance(validation["x"], support)
    test_distance_raw = support_distance(test["x"], support)
    distance_scale = positive_scale(val_distance_raw)
    val_distance = val_distance_raw / distance_scale
    test_distance = test_distance_raw / distance_scale

    fitted = []
    for config in CONFIGS:
        print(f"{name}: fitting {config}", flush=True)
        row = fit_configuration(
            model_train, checkpoint, validation, test, config
        )
        row["authority"] = calibrate_arms(
            validation, row["arrays"]["validation"], val_distance
        )
        primary = row["authority"]["monotone_shell_authority"]
        row["primary_oof_group_mse"] = float(np.mean(group_mse(
            validation["y"], primary.oof_prediction, validation["groups"]
        )))
        fitted.append(row)
    selected = min(
        fitted,
        key=lambda row: (
            not row["authority"]["monotone_shell_authority"].approved,
            row["primary_oof_group_mse"],
            row["config"]["width"],
        ),
    )

    validation_components = selected["arrays"]["validation"]
    test_components = selected["arrays"]["test"]
    arms = {
        "fallback": test_components["fallback"],
        "prior_only": test_components["prior"],
        "full_residual": test_components["candidate"],
    }
    authority_rows = {}
    for arm_name, result in selected["authority"].items():
        arms[arm_name] = apply_residual_authority(
            test_components["prior"],
            test_components["candidate"],
            test_components["fallback"],
            test_distance,
            result.policy,
        )
        authority_rows[arm_name] = {
            "approved": result.approved,
            "policy": asdict(result.policy),
            "oof_relative_gain_vs_fallback":
                result.oof_relative_gain_vs_fallback,
            "oof_mean_harm_ratio": result.oof_mean_harm_ratio,
            "oof_tail_harm_ratio": result.oof_tail_harm_ratio,
            "fold_authorities": result.fold_authorities,
        }

    evaluated = {}
    for arm_name, prediction in arms.items():
        evaluated[arm_name] = {
            "test": regression_metrics(
                test["y"], prediction, test["groups"]
            ),
            "test_harm": harm_ratios(
                test["y"], test["groups"], arms["fallback"], prediction
            ),
        }
    payload = {
        "split_ids": ids,
        "inner_split_ids": inner_ids,
        "hull": hull,
        "distance": {
            "scale": distance_scale,
            "validation_mean": float(np.mean(val_distance)),
            "test_mean": float(np.mean(test_distance)),
        },
        "candidate_configurations": [
            {
                "config": row["config"],
                "epochs": row["epochs"],
                "primary_approved": row["authority"][
                    "monotone_shell_authority"
                ].approved,
                "primary_oof_group_mse": row["primary_oof_group_mse"],
                "primary_oof_gain": row["authority"][
                    "monotone_shell_authority"
                ].oof_relative_gain_vs_fallback,
            }
            for row in fitted
        ],
        "selected_config": selected["config"],
        "authority": authority_rows,
        "arms": evaluated,
    }
    return payload, arms


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite residual-authority run")
    results, predictions = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        results[name], arms = evaluate(name, prepared)
        predictions.update({
            f"{name}_{arm}": prediction
            for arm, prediction in arms.items()
        })
        print(name, {
            arm: round(row["test"]["pooled"]["r2"], 4)
            for arm, row in results[name]["arms"].items()
        }, flush=True)
    result_path.write_text(json.dumps({
        "status": (
            "retrospective residual-authority development; "
            "not prospective confirmation"
        ),
        "protocol": PROTOCOL,
        "seeds": SEEDS,
        "configs": CONFIGS,
        "authority_settings": AUTHORITY_SETTINGS,
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **predictions)


if __name__ == "__main__":
    main()
