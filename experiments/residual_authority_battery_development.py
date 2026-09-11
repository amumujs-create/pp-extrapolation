#!/usr/bin/env python3
"""Residual-authority ablation on the matched battery prior/residual controls."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PAE = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(PAE), str(ROOT / "experiments")]

from boundary_quotient_pp_batteries import build_rows, full_part, score_by_dataset
from bq_pp_matched_controls import (
    ALPHA,
    BOUND,
    LR,
    WD,
    WIDTH,
    fit_direct,
    predict_direct,
)
from pae_boundary_realdata import DATASETS, prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale, concatenate_rows
from pp_extrapolation.boundary_quotient import (
    fit_boundary_quotient_pp,
    predict_boundary_affine,
    predict_boundary_quotient,
)
from pp_extrapolation.residual_authority import (
    ResidualAuthorityPolicy,
    apply_residual_authority,
    crossfit_residual_authority,
    distance_shell_edges,
)
from pp_extrapolation.risk_budgeted_prior import group_mse

OUT = ROOT / "results" / "residual_authority_battery_development"
PROTOCOL = "protocols/RESIDUAL_AUTHORITY_BATTERY_DEVELOPMENT_PROTOCOL.md"
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


def inner_group_split(rows: dict) -> tuple[dict, dict, dict]:
    checkpoint_units = {}
    checkpoint_mask = np.zeros(len(rows["y"]), dtype=bool)
    for dataset in np.unique(rows["dataset"]):
        take = rows["dataset"] == dataset
        labels = np.asarray(sorted(np.unique(rows["units"][take]), key=str))
        selected = labels[4::5]
        if len(selected) < 2:
            selected = labels[-2:]
        checkpoint_units[str(int(dataset))] = [
            str(value) for value in selected
        ]
        checkpoint_mask |= take & np.isin(rows["units"], selected)
    return (
        subset(rows, ~checkpoint_mask),
        subset(rows, checkpoint_mask),
        checkpoint_units,
    )


def positive_scale(values: np.ndarray) -> float:
    values = np.asarray(values)
    positive = values[values > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def fit_models(model_train, checkpoint, full, validation, source):
    selection = {
        split: {"fallback": [], "prior": [], "candidate": []}
        for split in ("validation", "source")
    }
    epochs = {"fallback": [], "candidate": []}
    for seed in SEEDS:
        direct_selected = fit_direct(
            model_train, checkpoint, seed, 0.0, 500, 70
        )
        bq_selected = fit_boundary_quotient_pp(
            model_train,
            checkpoint,
            seed=seed,
            width=WIDTH,
            alpha=ALPHA,
            learning_rate=LR,
            weight_decay=WD,
            residual_bound=BOUND,
            max_epochs=500,
            patience=70,
        )
        direct_epochs = max(int(direct_selected["epoch"]), 1)
        bq_epochs = max(
            int(bq_selected.selection["selected_epoch"]), 1
        )
        epochs["fallback"].append(direct_epochs)
        epochs["candidate"].append(bq_epochs)
        selection["validation"]["fallback"].append(
            predict_direct(direct_selected, validation)
        )
        selection["validation"]["prior"].append(
            predict_boundary_affine(bq_selected, validation)
        )
        selection["validation"]["candidate"].append(
            predict_boundary_quotient(bq_selected, validation)
        )

        direct_final = fit_direct(
            full, full, seed, 0.0, direct_epochs, 10000, False
        )
        bq_final = fit_boundary_quotient_pp(
            full,
            full,
            seed=seed,
            width=WIDTH,
            alpha=ALPHA,
            learning_rate=LR,
            weight_decay=WD,
            residual_bound=BOUND,
            max_epochs=bq_epochs,
            patience=10000,
            restore_best=False,
        )
        selection["source"]["fallback"].append(
            predict_direct(direct_final, source)
        )
        selection["source"]["prior"].append(
            predict_boundary_affine(bq_final, source)
        )
        selection["source"]["candidate"].append(
            predict_boundary_quotient(bq_final, source)
        )
        print(f"seed {seed} fitted", flush=True)
    return {
        "epochs": epochs,
        "components": {
            split: {
                name: np.mean(np.asarray(values), axis=0)
                for name, values in rows.items()
            }
            for split, rows in selection.items()
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


def calibrate_dataset(
    validation,
    source,
    val_components,
    source_components,
    dataset,
):
    val_mask = validation["dataset"] == dataset
    source_mask = source["dataset"] == dataset
    scale = positive_scale(validation["margin"][val_mask])
    val_distance = validation["margin"][val_mask] / scale
    source_distance = source["margin"][source_mask] / scale
    edges = distance_shell_edges(val_distance)
    common = {
        "y": validation["y"][val_mask],
        "groups": validation["units"][val_mask],
        "fallback": val_components["fallback"][val_mask],
        "prior": val_components["prior"][val_mask],
        "candidate": val_components["candidate"][val_mask],
        "distance": val_distance,
        **AUTHORITY_SETTINGS,
    }
    enough_units = len(np.unique(common["groups"])) >= 5
    results = {}
    if enough_units:
        results = {
            "global_authority": crossfit_residual_authority(
                **common, edges=(0.0, float("inf"))
            ),
            "free_shell_authority": crossfit_residual_authority(
                **common, edges=edges, enforce_monotone=False
            ),
            "monotone_shell_authority": crossfit_residual_authority(
                **common, edges=edges, enforce_monotone=True
            ),
        }
    predictions = {
        "fallback": source_components["fallback"][source_mask],
        "prior_only": source_components["prior"][source_mask],
        "full_residual": source_components["candidate"][source_mask],
    }
    authority = {}
    for name in (
        "global_authority",
        "free_shell_authority",
        "monotone_shell_authority",
    ):
        if enough_units:
            result = results[name]
            policy = result.policy
            authority[name] = {
                "approved": result.approved,
                "reason": (
                    "cross-fitted evidence"
                    if result.approved else "cross-fitted evidence rejected"
                ),
                "policy": asdict(policy),
                "oof_relative_gain_vs_fallback":
                    result.oof_relative_gain_vs_fallback,
                "oof_mean_harm_ratio": result.oof_mean_harm_ratio,
                "oof_tail_harm_ratio": result.oof_tail_harm_ratio,
                "fold_authorities": result.fold_authorities,
            }
        else:
            arm_edges = (
                (0.0, float("inf"))
                if name == "global_authority" else edges
            )
            policy = ResidualAuthorityPolicy(
                edges=arm_edges,
                authorities=(None,) * (len(arm_edges) - 1),
                validation_loss=float(np.mean(group_mse(
                    common["y"], common["fallback"], common["groups"]
                ))),
                relative_gain_vs_fallback=0.0,
                mean_harm_ratio=0.0,
                tail_harm_ratio=0.0,
                active_shells=0,
            )
            authority[name] = {
                "approved": False,
                "reason": "fewer than five physical validation units",
                "policy": asdict(policy),
                "oof_relative_gain_vs_fallback": None,
                "oof_mean_harm_ratio": None,
                "oof_tail_harm_ratio": None,
                "fold_authorities": (),
            }
        predictions[name] = apply_residual_authority(
            predictions["prior_only"],
            predictions["full_residual"],
            predictions["fallback"],
            source_distance,
            policy,
        )
    truth = source["y"][source_mask]
    groups = source["units"][source_mask]
    metrics = {}
    for name, prediction in predictions.items():
        metrics[name] = {
            "pooled_r2": float(
                1 - np.sum((truth - prediction) ** 2)
                / np.sum((truth - np.mean(truth)) ** 2)
            ),
            "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
            "harm": harm_ratios(
                truth, groups, predictions["fallback"], prediction
            ),
        }
    return {
        "distance_scale": scale,
        "validation_shell_edges": edges,
        "authority": authority,
        "test": metrics,
    }, predictions


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite battery authority run")
    scales, parts = {}, {
        key: [] for key in ("train", "validation", "full", "source")
    }
    audits = {}
    for index, name in enumerate(DATASETS):
        split, audit = prepare_dataset(name)
        audits[name] = audit
        scales[name] = BatteryRepresentationScale.fit(
            split["train"], audit["boundary"]
        )
        for key, part in (
            ("train", split["train"]),
            ("validation", split["val"]),
            ("full", full_part(split)),
            ("source", split["source"]),
        ):
            parts[key].append(build_rows(part, scales[name], index))
    rows = {
        key: concatenate_rows(value)
        for key, value in parts.items()
    }
    model_train, checkpoint, inner_units = inner_group_split(rows["train"])
    fitted = fit_models(
        model_train,
        checkpoint,
        rows["full"],
        rows["validation"],
        rows["source"],
    )
    results, arrays = {}, {}
    for index, name in enumerate(DATASETS):
        results[name], predictions = calibrate_dataset(
            rows["validation"],
            rows["source"],
            fitted["components"]["validation"],
            fitted["components"]["source"],
            index,
        )
        for arm, prediction in predictions.items():
            arrays[f"{name}_{arm}"] = prediction
        print(name, {
            arm: round(values["pooled_r2"], 4)
            for arm, values in results[name]["test"].items()
        }, flush=True)
    target.write_text(json.dumps({
        "status": (
            "retrospective boundary-quotient residual-authority development; "
            "not prospective confirmation"
        ),
        "protocol": PROTOCOL,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "shared_budget": {
            "width": WIDTH,
            "alpha": ALPHA,
            "learning_rate": LR,
            "weight_decay": WD,
            "residual_bound": BOUND,
        },
        "authority_settings": AUTHORITY_SETTINGS,
        "inner_checkpoint_units": inner_units,
        "selected_epochs": fitted["epochs"],
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
