#!/usr/bin/env python3
"""Retrospective matched audit of the frozen paper-level PP-X policy.

The common PP and matched-direct predictions are already stored on identical
source/validation/test splits. Test outcomes are used only after each policy
decision has been computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.paper_ppx import (  # noqa: E402
    PPXCandidateEvidence,
    PPXContract,
    select_paper_ppx,
)
from pp_extrapolation.transferability_gate import PriorEvidence  # noqa: E402

SOURCE = ROOT / "results/cross_domain_mechanism_v2"
OUT = ROOT / "results/ppx_paper_policy_audit_v1"
RNG = np.random.default_rng(20260911)


def unit_rmse(y, groups, prediction):
    return np.asarray([
        np.sqrt(np.mean((y[groups == unit] - prediction[groups == unit]) ** 2))
        for unit in np.unique(groups)
    ])


def validation_evidence(saved):
    y = np.asarray(saved["validation_y"], float)
    groups = np.asarray(saved["validation_groups"])
    direct = np.asarray(saved["validation_plain"], float).mean(0)
    pp = np.asarray(saved["validation_pp"], float).mean(0)
    direct_unit = unit_rmse(y, groups, direct)
    pp_unit = unit_rmse(y, groups, pp)
    return {
        "direct_loss": float(np.mean((y - direct) ** 2)),
        "pp_loss": float(np.mean((y - pp) ** 2)),
        "unit_win_fraction": float(np.mean(pp_unit < direct_unit)),
        "worst_unit_ratio": float(np.max(
            pp_unit / np.maximum(direct_unit, 1e-12)
        )),
    }


def test_effect(saved):
    y = np.asarray(saved["y"], float)
    groups = np.asarray(saved["groups"])
    direct = np.asarray(saved["plain"], float).mean(0)
    pp = np.asarray(saved["pp"], float).mean(0)
    direct_unit = unit_rmse(y, groups, direct)
    pp_unit = unit_rmse(y, groups, pp)
    relative = 1.0 - pp_unit / np.maximum(direct_unit, 1e-12)
    log_ratio = np.log(
        np.maximum(direct_unit, 1e-12) / np.maximum(pp_unit, 1e-12)
    )
    return {
        "pp_helped": bool(np.mean(relative) > 0),
        "mean_relative_unit_gain": float(np.mean(relative)),
        "mean_unit_log_rmse_ratio": float(np.mean(log_ratio)),
        "pp_unit_wins": int(np.sum(relative > 0)),
        "n_units": len(relative),
    }


def summarize(rows, field):
    selected = np.asarray([row[field] for row in rows], bool)
    actual = np.asarray([row["test"]["pp_helped"] for row in rows], bool)
    effects = np.asarray([
        row["test"]["mean_unit_log_rmse_ratio"] if choose else 0.0
        for row, choose in zip(rows, selected)
    ])
    index = RNG.integers(0, len(rows), size=(50_000, len(rows)))
    boot = effects[index].mean(1)
    return {
        "accepted": int(np.sum(selected)),
        "correct": int(np.sum(selected == actual)),
        "accuracy": float(np.mean(selected == actual)),
        "false_accepts": int(np.sum(selected & ~actual)),
        "false_rejects": int(np.sum(~selected & actual)),
        "dataset_improvements": int(np.sum(effects > 0)),
        "dataset_losses": int(np.sum(effects < 0)),
        "equal_dataset_mean_log_rmse_ratio": float(np.mean(effects)),
        "bootstrap_ci95": np.quantile(boot, (0.025, 0.975)).tolist(),
    }


def main():
    metadata = json.loads((SOURCE / "results.json").read_text())
    rows = []
    contract = PPXContract(
        known_boundary=True,
        ordered_progression=True,
        causal_history=False,
        observed_regime=False,
        support_heterogeneity_available=False,
        fallback="direct_fallback",
    )
    # This audit isolates validation approval. Prior admissibility is held true;
    # the separate contract audit evaluates whether a prior may enter this stage.
    prior = PriorEvidence(
        known_boundary=True,
        complete_groups=0,
        minimum_complete_groups_per_regime=0,
    )
    for name in metadata["datasets"]:
        saved = np.load(
            SOURCE / f"{name}_predictions.npz", allow_pickle=True
        )
        validation = validation_evidence(saved)
        decision = select_paper_ppx(
            contract,
            prior,
            (
                PPXCandidateEvidence(
                    "direct_fallback",
                    validation["direct_loss"],
                    0.0,
                    1.0,
                ),
                PPXCandidateEvidence(
                    "unbounded",
                    validation["pp_loss"],
                    validation["unit_win_fraction"],
                    validation["worst_unit_ratio"],
                ),
            ),
        )
        test = test_effect(saved)
        rows.append({
            "dataset": name,
            "validation": validation,
            "test": test,
            "paper_policy_accepts_pp": decision.executor == "unbounded",
            "simple_validation_accepts_pp": (
                validation["pp_loss"] < validation["direct_loss"]
            ),
            "always_pp": True,
            "always_direct": False,
            # Nondeployable reference, computed only after decisions above.
            "test_oracle": test["pp_helped"],
            "decision": {
                "executor": decision.executor,
                "reason": decision.reason,
            },
        })
    payload = {
        "status": "retrospective common-backbone paper-policy audit",
        "owner": "박진서",
        "holdouts_loaded": False,
        "frozen_policy": {
            "minimum_validation_gain": 0.02,
            "minimum_unit_win_fraction": 0.60,
            "maximum_worst_unit_rmse_ratio": 1.10,
        },
        "policies": {
            "always_direct": summarize(rows, "always_direct"),
            "always_pp": summarize(rows, "always_pp"),
            "simple_validation": summarize(
                rows, "simple_validation_accepts_pp"
            ),
            "paper_ppx": summarize(rows, "paper_policy_accepts_pp"),
            "test_oracle_non_deployable": summarize(rows, "test_oracle"),
        },
        "datasets": rows,
        "guardrails": [
            "Policy decisions are computed before test arrays are scored.",
            "This is retrospective and cannot validate prospective transfer.",
            "The audit isolates executor approval with prior admissibility held true.",
            "The test oracle is an upper bound and is not a deployable comparator.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["policies"], indent=2))


if __name__ == "__main__":
    main()
