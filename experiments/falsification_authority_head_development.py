#!/usr/bin/env python3
"""LODO learned regret-bound residual authority experiment."""
from __future__ import annotations

import json
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.falsification_authority_head import (  # noqa: E402
    apply_unit_authority,
    fit_falsification_authority_head,
    predict_residual_authority,
)
from pp_extrapolation.prior_falsification import (  # noqa: E402
    certify_prior_falsification,
    unit_log_regret,
)

SOURCE = ROOT / "results" / "cross_domain_mechanism_v2"
OUT = ROOT / "results" / "falsification_authority_head_v1"
PROTOCOL = "protocols/FALSIFICATION_AWARE_AUTHORITY_HEAD_PROTOCOL.md"


def unit_features(groups, fallback_seeds, candidate_seeds, distance):
    groups = np.asarray(groups).astype(str)
    fallback_seeds = np.asarray(fallback_seeds, dtype=np.float64)
    candidate_seeds = np.asarray(candidate_seeds, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    labels = np.unique(groups)
    rows = []
    for label in labels:
        mask = groups == label
        fallback = np.mean(fallback_seeds[:, mask], axis=0)
        candidate = np.mean(candidate_seeds[:, mask], axis=0)
        correction = candidate - fallback
        scale = float(np.quantile(fallback, 0.75) - np.quantile(fallback, 0.25))
        if scale <= 1e-12:
            scale = max(float(np.std(fallback)), 1.0)
        local_distance = np.log1p(distance[mask])
        fallback_disagreement = np.std(fallback_seeds[:, mask], axis=0)
        candidate_disagreement = np.std(candidate_seeds[:, mask], axis=0)
        fallback_range = max(float(np.ptp(fallback)), 1e-12)
        candidate_range = float(np.ptp(candidate))
        rows.append([
            np.log1p(np.sum(mask)),
            float(np.mean(local_distance)),
            float(np.quantile(local_distance, 0.90)),
            float(np.mean(distance[mask] > 0)),
            float(np.mean(fallback_disagreement) / scale),
            float(np.mean(candidate_disagreement) / scale),
            float(np.mean(np.abs(correction)) / scale),
            float(np.quantile(np.abs(correction), 0.90) / scale),
            float(np.mean(correction) / scale),
            float(np.log1p(candidate_range / fallback_range)),
        ])
    return labels, np.asarray(rows, dtype=np.float64)


def load_domain(name):
    saved = np.load(
        SOURCE / f"{name}_predictions.npz", allow_pickle=True
    )
    validation_groups = np.asarray(saved["validation_groups"]).astype(str)
    validation_labels, validation_x = unit_features(
        validation_groups,
        saved["validation_plain"],
        saved["validation_pp"],
        saved["validation_distance"],
    )
    validation_y = np.asarray(saved["validation_y"], dtype=np.float64)
    validation_fallback = np.asarray(
        saved["validation_plain"], dtype=np.float64
    ).mean(0)
    validation_candidate = np.asarray(
        saved["validation_pp"], dtype=np.float64
    ).mean(0)
    validation_regret = unit_log_regret(
        validation_y,
        validation_groups,
        validation_fallback,
        validation_candidate,
    )
    test_groups = np.asarray(saved["groups"]).astype(str)
    test_labels, test_x = unit_features(
        test_groups,
        saved["plain"],
        saved["pp"],
        saved["test_distance"],
    )
    certificate = certify_prior_falsification(
        validation_y,
        validation_groups,
        validation_fallback,
        validation_candidate,
        seed=20260912 + zlib.crc32(name.encode()),
    )
    return {
        "validation_labels": validation_labels,
        "validation_features": validation_x,
        "validation_regret": validation_regret,
        "test_labels": test_labels,
        "test_features": test_x,
        "test_y": np.asarray(saved["y"], dtype=np.float64),
        "test_groups": test_groups,
        "test_fallback": np.asarray(saved["plain"], dtype=np.float64).mean(0),
        "test_candidate": np.asarray(saved["pp"], dtype=np.float64).mean(0),
        "certificate": certificate,
    }


def score(y, groups, fallback, prediction):
    improvement = -unit_log_regret(y, groups, fallback, prediction)
    return {
        "mean_unit_log_rmse_improvement": float(np.mean(improvement)),
        "unit_wins": int(np.sum(improvement > 0)),
        "n_units": len(improvement),
    }


def summarize(rows, arm, seed):
    effect = np.asarray([
        row["arms"][arm]["mean_unit_log_rmse_improvement"] for row in rows
    ])
    active = np.asarray([row["arms"][arm]["active"] for row in rows], bool)
    actual = np.asarray([
        row["arms"]["always_pp"]["mean_unit_log_rmse_improvement"] > 0
        for row in rows
    ])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(effect), size=(50_000, len(effect)))
    bootstrap = np.mean(effect[index], axis=1)
    return {
        "active_domains": int(np.sum(active)),
        "decision_accuracy": float(np.mean(active == actual)),
        "false_harm_domains": int(np.sum(effect < -1e-12)),
        "improved_domains": int(np.sum(effect > 1e-12)),
        "equal_domain_mean_unit_log_rmse_improvement": float(np.mean(effect)),
        "bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def main():
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite authority-head experiment")
    metadata = json.loads((SOURCE / "results.json").read_text())
    names = list(metadata["datasets"])
    data = {name: load_domain(name) for name in names}

    # Every held-out test domain is predicted by a head trained only on the
    # other domains' validation units.
    rows = []
    predictions = {}
    for held_out in names:
        training_names = [name for name in names if name != held_out]
        features = np.concatenate([
            data[name]["validation_features"] for name in training_names
        ])
        regret = np.concatenate([
            data[name]["validation_regret"] for name in training_names
        ])
        domains = np.concatenate([
            np.repeat(name, len(data[name]["validation_regret"]))
            for name in training_names
        ])
        model = fit_falsification_authority_head(features, regret, domains)
        held = data[held_out]
        authority, upper = predict_residual_authority(
            model, held["test_features"]
        )
        binary = (upper < 0).astype(np.float64)
        certificate_authority = np.full(
            len(held["test_labels"]),
            float(not held["certificate"].falsified),
        )
        arm_authority = {
            "direct_fallback": np.zeros(len(authority)),
            "always_pp": np.ones(len(authority)),
            "within_domain_certificate": certificate_authority,
            "learned_binary_authority": binary,
            "learned_continuous_authority": authority,
        }
        arms = {}
        for arm, values in arm_authority.items():
            prediction = apply_unit_authority(
                held["test_fallback"],
                held["test_candidate"],
                held["test_groups"],
                held["test_labels"],
                values,
            )
            predictions[f"{held_out}_{arm}"] = prediction
            arms[arm] = {
                **score(
                    held["test_y"],
                    held["test_groups"],
                    held["test_fallback"],
                    prediction,
                ),
                "active": bool(np.any(values > 0)),
                "mean_authority": float(np.mean(values)),
                "active_unit_fraction": float(np.mean(values > 0)),
            }
        oracle_values = []
        for label in held["test_labels"]:
            mask = held["test_groups"] == label
            base_error = np.mean(
                (held["test_fallback"][mask] - held["test_y"][mask]) ** 2
            )
            trial_error = np.mean(
                (held["test_candidate"][mask] - held["test_y"][mask]) ** 2
            )
            oracle_values.append(float(trial_error < base_error))
        oracle_prediction = apply_unit_authority(
            held["test_fallback"],
            held["test_candidate"],
            held["test_groups"],
            held["test_labels"],
            np.asarray(oracle_values),
        )
        arms["unit_oracle_non_deployable"] = {
            **score(
                held["test_y"],
                held["test_groups"],
                held["test_fallback"],
                oracle_prediction,
            ),
            "active": bool(np.any(oracle_values)),
            "mean_authority": float(np.mean(oracle_values)),
            "active_unit_fraction": float(np.mean(np.asarray(oracle_values) > 0)),
        }
        rows.append({
            "dataset": held_out,
            "head": {
                **asdict(model),
                "center": model.center.tolist(),
                "scale": model.scale.tolist(),
                "coefficient": model.coefficient.tolist(),
            },
            "predicted_upper": upper.tolist(),
            "continuous_authority": authority.tolist(),
            "within_domain_certificate": asdict(held["certificate"]),
            "arms": arms,
        })

    arm_names = tuple(rows[0]["arms"])
    payload = {
        "status": "opened retrospective leave-one-domain-out development",
        "owner": "박진서",
        "protocol": PROTOCOL,
        "feature_names": [
            "log_row_count",
            "mean_log_support_distance",
            "q90_log_support_distance",
            "outside_support_fraction",
            "direct_seed_disagreement",
            "pp_seed_disagreement",
            "mean_correction_magnitude",
            "q90_correction_magnitude",
            "signed_correction_mean",
            "log_prediction_range_ratio",
        ],
        "policies": {
            arm: summarize(rows, arm, 20260912 + index)
            for index, arm in enumerate(arm_names)
        },
        "datasets": rows,
        "guardrails": [
            "The held-out domain contributes no labels to its head.",
            "Only outcome-free test features determine test authority.",
            "Inner upper-bound residuals are leave-one-domain-out.",
            "The unit oracle is nondeployable.",
            "This remains retrospective because all domains were previously opened.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **predictions)
    print(json.dumps(payload["policies"], indent=2))


if __name__ == "__main__":
    main()
