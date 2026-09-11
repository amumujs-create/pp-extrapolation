#!/usr/bin/env python3
"""Fixed prior-falsification certificate audit on 12 opened datasets."""
from __future__ import annotations

import json
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.prior_falsification import (  # noqa: E402
    certify_prior_falsification,
    unit_log_regret,
)

SOURCE = ROOT / "results" / "cross_domain_mechanism_v2"
OUT = ROOT / "results" / "prior_falsification_certificate_v1"
PROTOCOL = "protocols/PRIOR_FALSIFICATION_CERTIFICATE_PROTOCOL.md"
BOOTSTRAP_REPLICATES = 20_000


def unit_rmse(y, groups, prediction):
    return np.asarray([
        np.sqrt(np.mean((y[groups == unit] - prediction[groups == unit]) ** 2))
        for unit in np.unique(groups)
    ])


def validation_evidence(saved, dataset):
    y = np.asarray(saved["validation_y"], dtype=np.float64)
    groups = np.asarray(saved["validation_groups"])
    fallback = np.asarray(saved["validation_plain"], dtype=np.float64).mean(0)
    candidate = np.asarray(saved["validation_pp"], dtype=np.float64).mean(0)
    base_unit = unit_rmse(y, groups, fallback)
    trial_unit = unit_rmse(y, groups, candidate)
    certificate = certify_prior_falsification(
        y,
        groups,
        fallback,
        candidate,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=20260912 + zlib.crc32(dataset.encode()),
    )
    base_loss = float(np.mean((y - fallback) ** 2))
    trial_loss = float(np.mean((y - candidate) ** 2))
    paper = (
        trial_loss < 0.98 * base_loss
        and float(np.mean(trial_unit < base_unit)) >= 0.60
        and float(np.max(trial_unit / np.maximum(base_unit, 1e-12))) <= 1.10
    )
    return {
        "fallback_loss": base_loss,
        "candidate_loss": trial_loss,
        "relative_mse_gain": (base_loss - trial_loss) / max(base_loss, 1e-12),
        "unit_win_fraction": float(np.mean(trial_unit < base_unit)),
        "worst_unit_rmse_ratio": float(np.max(
            trial_unit / np.maximum(base_unit, 1e-12)
        )),
        "certificate": asdict(certificate),
        "decisions": {
            "always_direct": False,
            "always_pp": True,
            "simple_validation": trial_loss < base_loss,
            "paper_ppx": paper,
            "certificate_only": not certificate.falsified,
            "paper_plus_falsification": paper and not certificate.falsified,
        },
    }


def test_effect(saved):
    y = np.asarray(saved["y"], dtype=np.float64)
    groups = np.asarray(saved["groups"])
    fallback = np.asarray(saved["plain"], dtype=np.float64).mean(0)
    candidate = np.asarray(saved["pp"], dtype=np.float64).mean(0)
    regret = unit_log_regret(y, groups, fallback, candidate)
    log_improvement = -regret
    return {
        "pp_helped": bool(float(np.mean(log_improvement)) > 0),
        "mean_unit_log_rmse_improvement": float(np.mean(log_improvement)),
        "pp_unit_wins": int(np.sum(log_improvement > 0)),
        "n_units": len(log_improvement),
    }


def summarize(rows, policy, seed):
    selected = np.asarray([
        row["validation"]["decisions"][policy] for row in rows
    ], dtype=bool)
    actual = np.asarray([row["test"]["pp_helped"] for row in rows], dtype=bool)
    effect = np.asarray([
        row["test"]["mean_unit_log_rmse_improvement"] if choose else 0.0
        for row, choose in zip(rows, selected)
    ])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(rows), size=(50_000, len(rows)))
    bootstrap = np.mean(effect[index], axis=1)
    return {
        "accepted": int(np.sum(selected)),
        "correct": int(np.sum(selected == actual)),
        "accuracy": float(np.mean(selected == actual)),
        "false_accepts": int(np.sum(selected & ~actual)),
        "false_rejects": int(np.sum(~selected & actual)),
        "accepted_wins": int(np.sum(selected & actual)),
        "accepted_losses": int(np.sum(selected & ~actual)),
        "equal_dataset_mean_log_rmse_improvement": float(np.mean(effect)),
        "bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def main():
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite falsification certificate run")
    metadata = json.loads((SOURCE / "results.json").read_text())
    names = list(metadata["datasets"])

    # Materialize every decision from validation evidence before test scoring.
    validation = {}
    for name in names:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        validation[name] = validation_evidence(saved, name)

    rows = []
    for name in names:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        rows.append({
            "dataset": name,
            "validation": validation[name],
            "test": test_effect(saved),
        })
    for row in rows:
        row["validation"]["decisions"]["test_oracle_non_deployable"] = (
            row["test"]["pp_helped"]
        )

    policies = (
        "always_direct",
        "always_pp",
        "simple_validation",
        "paper_ppx",
        "certificate_only",
        "paper_plus_falsification",
        "test_oracle_non_deployable",
    )
    payload = {
        "status": "opened retrospective development; not confirmation",
        "owner": "박진서",
        "protocol": PROTOCOL,
        "certificate": {
            "effect": "unit log(RMSE_candidate / RMSE_fallback)",
            "confidence": 0.95,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "threshold": "one-sided upper confidence bound < 0",
        },
        "policies": {
            policy: summarize(rows, policy, 20260912 + index)
            for index, policy in enumerate(policies)
        },
        "datasets": rows,
        "guardrails": [
            "No threshold grid was searched.",
            "All policy decisions were materialized before test scoring.",
            "Physical units, not rows or seeds, were bootstrapped.",
            "The test oracle is nondeployable.",
            "A new untouched domain is required for confirmation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["policies"], indent=2))


if __name__ == "__main__":
    main()
