#!/usr/bin/env python3
"""LODO audit for Cross-Domain Consensus Residual PP-X."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from final_modular_pp_evidence import build_datasets, r2  # noqa: E402
from pp_extrapolation.cross_domain_consensus import (  # noqa: E402
    consensus_representation,
    fit_cross_domain_consensus_head,
    normalized_residual_target,
    predict_cross_domain_consensus,
)

OUT = ROOT / "results" / "cross_domain_consensus_residual_ppx_v1"
PROTOCOL = "protocols/CROSS_DOMAIN_CONSENSUS_RESIDUAL_PPX_PROTOCOL.md"


def prepare():
    rows = []
    for name, (y, groups, predictions, *_unused) in build_datasets().items():
        y = np.asarray(y, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        representation = consensus_representation(predictions)
        rows.append({
            "dataset": name,
            "y": y,
            "groups": np.asarray(groups).astype(str),
            "predictions": predictions,
            "representation": representation,
            "residual_target": normalized_residual_target(y, representation),
        })
    return rows


def fit_other_domains(rows, held_out):
    training = [row for row in rows if row["dataset"] != held_out]
    features = np.concatenate([
        row["representation"].features for row in training
    ])
    target = np.concatenate([row["residual_target"] for row in training])
    domains = np.concatenate([
        np.repeat(row["dataset"], len(row["y"])) for row in training
    ])
    return fit_cross_domain_consensus_head(features, target, domains)


def main():
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite consensus PP-X audit")
    rows = prepare()
    results = []
    arrays = {}
    base_sse = updated_sse = total_variation = 0.0
    all_base_seed_r2 = []
    all_updated_seed_r2 = []
    for row in rows:
        model = fit_other_domains(rows, row["dataset"])
        prediction = predict_cross_domain_consensus(
            model, row["predictions"]
        )
        y = row["y"]
        base_ensemble = row["representation"].ensemble
        base_seed_r2 = np.asarray([
            r2(y, value) for value in row["predictions"]
        ])
        updated_seed_r2 = np.asarray([
            r2(y, value) for value in prediction.seeds
        ])
        base_r2 = r2(y, base_ensemble)
        updated_r2 = r2(y, prediction.ensemble)
        base_sse += float(np.sum((y - base_ensemble) ** 2))
        updated_sse += float(np.sum((y - prediction.ensemble) ** 2))
        total_variation += float(np.sum((y - np.mean(y)) ** 2))
        all_base_seed_r2.extend(base_seed_r2)
        all_updated_seed_r2.extend(updated_seed_r2)
        arrays[f"{row['dataset']}_y"] = y
        arrays[f"{row['dataset']}_base_seeds"] = row["predictions"]
        arrays[f"{row['dataset']}_updated_seeds"] = prediction.seeds
        results.append({
            "dataset": row["dataset"],
            "n_rows": len(y),
            "n_units": len(np.unique(row["groups"])),
            "active_meta_residual": prediction.active_residual,
            "q90_seed_disagreement": prediction.q90_seed_disagreement,
            "base_ensemble_r2": base_r2,
            "updated_ensemble_r2": updated_r2,
            "ensemble_r2_gain": updated_r2 - base_r2,
            "base_seed_r2": base_seed_r2.tolist(),
            "updated_seed_r2": updated_seed_r2.tolist(),
            "base_seed_r2_sd": float(np.std(base_seed_r2, ddof=1)),
            "updated_seed_r2_sd": float(np.std(updated_seed_r2, ddof=1)),
            "base_min_seed_r2": float(np.min(base_seed_r2)),
            "updated_min_seed_r2": float(np.min(updated_seed_r2)),
            "head": {
                "ridge_alpha": model.ridge_alpha,
                "training_domains": list(model.training_domains),
                "intercept": model.intercept,
                "coefficient": model.coefficient.tolist(),
            },
        })

    base_dataset = np.asarray([row["base_ensemble_r2"] for row in results])
    updated_dataset = np.asarray([
        row["updated_ensemble_r2"] for row in results
    ])
    base_seed = np.asarray(all_base_seed_r2)
    updated_seed = np.asarray(all_updated_seed_r2)
    rng = np.random.default_rng(20260913)
    delta = updated_dataset - base_dataset
    index = rng.integers(0, len(delta), size=(50_000, len(delta)))
    summary = {
        "dataset_positive_r2_base": int(np.sum(base_dataset > 0)),
        "dataset_positive_r2_updated": int(np.sum(updated_dataset > 0)),
        "seed_dataset_positive_r2_base": int(np.sum(base_seed > 0)),
        "seed_dataset_positive_r2_updated": int(np.sum(updated_seed > 0)),
        "dataset_r2_wins": int(np.sum(delta > 1e-12)),
        "dataset_r2_ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "dataset_r2_losses": int(np.sum(delta < -1e-12)),
        "mean_dataset_r2_base": float(np.mean(base_dataset)),
        "mean_dataset_r2_updated": float(np.mean(updated_dataset)),
        "mean_dataset_r2_gain_bootstrap_ci95": [
            float(value) for value in np.quantile(
                np.mean(delta[index], axis=1), (0.025, 0.975)
            )
        ],
        "minimum_dataset_r2_base": float(np.min(base_dataset)),
        "minimum_dataset_r2_updated": float(np.min(updated_dataset)),
        "global_normalized_pooled_r2_base": float(
            1.0 - base_sse / total_variation
        ),
        "global_normalized_pooled_r2_updated": float(
            1.0 - updated_sse / total_variation
        ),
        "mean_seed_r2_base": float(np.mean(base_seed)),
        "mean_seed_r2_updated": float(np.mean(updated_seed)),
        "minimum_seed_r2_base": float(np.min(base_seed)),
        "minimum_seed_r2_updated": float(np.min(updated_seed)),
        "seed_pair_wins": int(np.sum(updated_seed > base_seed + 1e-12)),
        "seed_pair_losses": int(np.sum(updated_seed < base_seed - 1e-12)),
        "datasets_with_reduced_seed_r2_sd": int(sum(
            row["updated_seed_r2_sd"] < row["base_seed_r2_sd"]
            for row in results
        )),
    }
    payload = {
        "model": "Cross-Domain Consensus Residual PP-X (CDCR-PPX)",
        "status": "retrospective LODO development; hyperparameters are not independently confirmed",
        "protocol": PROTOCOL,
        "frozen_policy": {
            "ridge_alpha": 100.0,
            "q90_seed_disagreement_cutoff": 0.12,
            "residual_authority": 0.50,
            "seed_shrinkage": 0.50,
        },
        "summary": summary,
        "datasets": results,
        "guardrails": [
            "Each held-out dataset is excluded from its fitted residual head.",
            "Target-time features contain predictions only, not labels.",
            "Hyperparameters were selected on opened Stage-0 datasets.",
            "An unopened cohort is required for confirmation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
