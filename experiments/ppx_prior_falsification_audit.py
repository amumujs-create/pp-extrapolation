#!/usr/bin/env python3
"""Grid sweep for prior-falsification certificate variants on paper policy."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from pp_extrapolation.paper_ppx import (
    PPXCandidateEvidence,
    PPXContract,
    select_paper_ppx,
)
from pp_extrapolation.transferability_gate import PriorEvidence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cross_domain_mechanism_v2"
OUT = ROOT / "results/ppx_prior_falsification_v1"


a = np.random.default_rng(20260911)


def unit_rmse(y, groups, pred):
    return np.asarray(
        [
            np.sqrt(np.mean((y[groups == unit] - pred[groups == unit]) ** 2))
            for unit in np.unique(groups)
        ]
    )


def bootstrap_ci_low(values, repeats=20_000):
    values = np.asarray(values, float)
    if len(values) <= 1:
        return float("-inf")
    idx = a.integers(0, len(values), size=(repeats, len(values)))
    return float(np.quantile(values[idx].mean(1), 0.025))


def validation_evidence(npz):
    y = np.asarray(npz["validation_y"], float)
    groups = np.asarray(npz["validation_groups"])
    plain = np.asarray(npz["validation_plain"], float).mean(0)
    pp = np.asarray(npz["validation_pp"], float).mean(0)
    plain_unit = unit_rmse(y, groups, plain)
    pp_unit = unit_rmse(y, groups, pp)
    unit_gain = plain_unit - pp_unit
    return {
        "direct_loss": float(np.mean((y - plain) ** 2)),
        "pp_loss": float(np.mean((y - pp) ** 2)),
        "unit_win_fraction": float(np.mean(pp_unit < plain_unit)),
        "worst_unit_ratio": float(np.max(pp_unit / np.maximum(plain_unit, 1e-12))),
        "unit_gain_ci_low": bootstrap_ci_low(unit_gain),
    }


def test_effect(npz):
    y = np.asarray(npz["y"], float)
    groups = np.asarray(npz["groups"])
    plain = np.asarray(npz["plain"], float).mean(0)
    pp = np.asarray(npz["pp"], float).mean(0)
    plain_unit = unit_rmse(y, groups, plain)
    pp_unit = unit_rmse(y, groups, pp)
    relative = 1.0 - pp_unit / np.maximum(plain_unit, 1e-12)
    return {
        "pp_helped": bool(np.mean(relative) > 0),
        "mean_relative_unit_gain": float(np.mean(relative)),
        "mean_unit_log_rmse_ratio": float(np.mean(np.log(np.maximum(plain_unit, 1e-12) / np.maximum(pp_unit, 1e-12)))),
        "pp_unit_wins": int(np.sum(relative > 0)),
        "n_units": len(relative),
    }


def summarize(rows, key):
    selected = np.asarray([row[key] for row in rows], bool)
    actual = np.asarray([row["test_oracle"] for row in rows], bool)
    effects = np.asarray([
        row["test"]["mean_unit_log_rmse_ratio"] if choose else 0.0
        for row, choose in zip(rows, selected)
    ])
    samples = a.integers(0, len(rows), size=(50_000, len(rows)))
    boot = effects[samples].mean(1)
    return {
        "accepted": int(np.sum(selected)),
        "correct": int(np.sum(selected == actual)),
        "accuracy": float(np.mean(selected == actual)),
        "false_accepts": int(np.sum(selected & ~actual)),
        "false_rejects": int(np.sum(~selected & actual)),
        "dataset_improvements": int(np.sum(effects > 0)),
        "dataset_losses": int(np.sum(effects < 0)),
        "mean_unit_log_rmse_ratio": float(np.mean(effects)),
        "bootstrap_ci95": np.quantile(boot, (0.025, 0.975)).tolist(),
    }


def run_decision(policy, evidence, meta):
    decision = select_paper_ppx(
        meta["contract"],
        meta["prior"],
        (
            PPXCandidateEvidence("direct_fallback", evidence["direct_loss"], 0.0, 1.0),
            PPXCandidateEvidence(
                "unbounded",
                evidence["pp_loss"],
                evidence["unit_win_fraction"],
                evidence["worst_unit_ratio"],
                unit_gain_ci_low=evidence.get("unit_gain_ci_low", float("-inf")),
            ),
        ),
        min_relative_improvement=policy["min_relative_improvement"],
        min_unit_win_fraction=policy["min_unit_win_fraction"],
        max_worst_unit_rmse_ratio=policy["max_worst_unit_rmse_ratio"],
        min_unit_gain_ci_low=policy["min_unit_gain_ci_low"],
    )
    return decision.executor == "unbounded"


def main():
    metadata = json.loads((SOURCE / "results.json").read_text())
    names = list(metadata["datasets"].keys())
    contract = PPXContract(
        known_boundary=True,
        ordered_progression=True,
        causal_history=False,
        observed_regime=False,
        support_heterogeneity_available=False,
        fallback="direct_fallback",
    )
    prior = PriorEvidence(
        known_boundary=True,
        complete_groups=0,
        minimum_complete_groups_per_regime=0,
    )
    rows = []
    policies = []

    for name in names:
        npz = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        evidence = validation_evidence(npz)
        test = test_effect(npz)
        rows.append({
            "dataset": name,
            "validation": evidence,
            "test": test,
        })

    # Policy grid for falsification-aware variants
    grid = []
    for improvement in [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.10]:
        for win_frac in [0.40, 0.50, 0.60, 0.70, 0.80]:
            for max_ratio in [1.00, 1.03, 1.05, 1.08, 1.10, 1.15]:
                for ci_low in [-5.0, -2.0, 0.0, 1.0, 2.0, 5.0]:
                    grid.append({
                        "name": f"falsification_g{improvement:.2f}_w{win_frac:.2f}_r{max_ratio:.2f}_c{ci_low:.2f}",
                        "min_relative_improvement": improvement,
                        "min_unit_win_fraction": win_frac,
                        "max_worst_unit_rmse_ratio": max_ratio,
                        "min_unit_gain_ci_low": ci_low,
                    })

    outcomes = []
    for policy in grid:
        decorated = []
        for row in rows:
            accepted = run_decision(policy, row["validation"], {"contract": contract, "prior": prior})
            decorated.append({
                "dataset": row["dataset"],
                "test_oracle": row["test"]["pp_helped"],
                "paper_policy_accepts_pp": accepted,
                "test": row["test"],
                "validation": row["validation"],
            })
        summary = summarize(decorated, "paper_policy_accepts_pp")
        summary.update(
            name=policy["name"],
            min_relative_improvement=policy["min_relative_improvement"],
            min_unit_win_fraction=policy["min_unit_win_fraction"],
            max_worst_unit_rmse_ratio=policy["max_worst_unit_rmse_ratio"],
            min_unit_gain_ci_low=policy["min_unit_gain_ci_low"],
            policy_variant="prior_falsification",
        )
        outcomes.append(summary)

    outcomes.sort(key=lambda r: (-r["accuracy"], r["false_accepts"], -r["correct"]))
    best = outcomes[0]

    payload = {
        "status": "prior falsification audit on paper PP-X policy",
        "owner": "박진서",
        "grid_size": len(grid),
        "best": best,
        "policies": outcomes,
        "datasets": [
            {
                **row,
                "always_pp": True,
                "always_direct": False,
                "simple_validation_accepts_pp": (row["validation"]["pp_loss"] < row["validation"]["direct_loss"]),
                "test_oracle": row["test"]["pp_helped"],
            }
            for row in rows
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"best": best, "candidates": len(outcomes)}, indent=2))


if __name__ == "__main__":
    main()
