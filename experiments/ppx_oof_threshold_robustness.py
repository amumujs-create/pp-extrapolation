#!/usr/bin/env python3
"""OOF robustness audit for PP-X approval thresholds.

Purpose
-------
Justify operational thresholds τ=(gain, unit-win, worst-ratio) by
leave-one-validation-unit-out stability on development units — NOT by
maximizing held-out test accuracy.

For each threshold cell:
  • OOF executor win fraction  (among LOO folds that accept PP)
  • OOF mean / worst regret    (relative unit RMSE vs direct when policy runs)
  • Route stability            (LOO decision == full-validation decision)

Frozen default (2%, 60%, 1.10) is reported inside the grid; we do not retune
Algorithm 1 from test outcomes.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pp_extrapolation.paper_ppx import (  # noqa: E402
    PPXCandidateEvidence,
    PPXContract,
    select_paper_ppx,
)
from pp_extrapolation.transferability_gate import PriorEvidence  # noqa: E402

SOURCE = ROOT / "results/cross_domain_mechanism_v2"
OUT = ROOT / "results/ppx_oof_threshold_robustness_v1"

GAINS = (0.01, 0.02, 0.05)
WINS = (0.50, 0.60, 0.70)
WORSTS = (1.05, 1.10, 1.20)
FROZEN = (0.02, 0.60, 1.10)
MIN_VAL_UNITS = 3  # LOO needs a usable remainder


def unit_rmse(y, groups, prediction):
    return {
        unit: float(np.sqrt(np.mean((y[groups == unit] - prediction[groups == unit]) ** 2)))
        for unit in np.unique(groups)
    }


def evidence_from_units(y, groups, direct, pp, keep_units):
    mask = np.isin(groups, list(keep_units))
    if not np.any(mask):
        return None
    y_k, g_k = y[mask], groups[mask]
    d_k, p_k = direct[mask], pp[mask]
    direct_u = unit_rmse(y_k, g_k, d_k)
    pp_u = unit_rmse(y_k, g_k, p_k)
    units = list(direct_u)
    if not units:
        return None
    direct_loss = float(np.mean((y_k - d_k) ** 2))
    pp_loss = float(np.mean((y_k - p_k) ** 2))
    wins = [pp_u[u] < direct_u[u] for u in units]
    ratios = [pp_u[u] / max(direct_u[u], 1e-12) for u in units]
    return {
        "direct_loss": direct_loss,
        "pp_loss": pp_loss,
        "unit_win_fraction": float(np.mean(wins)),
        "worst_unit_ratio": float(np.max(ratios)),
        "n_units": len(units),
    }


def accepts_pp(evidence: dict, gain: float, win: float, worst: float) -> bool:
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
    decision = select_paper_ppx(
        contract,
        prior,
        (
            PPXCandidateEvidence(
                "direct_fallback", evidence["direct_loss"], 0.0, 1.0
            ),
            PPXCandidateEvidence(
                "unbounded",
                evidence["pp_loss"],
                evidence["unit_win_fraction"],
                evidence["worst_unit_ratio"],
            ),
        ),
        min_relative_improvement=gain,
        min_unit_win_fraction=win,
        max_worst_unit_rmse_ratio=worst,
        min_unit_gain_ci_low=float("-inf"),
    )
    return decision.executor == "unbounded"


def relative_regret(direct_rmse: float, pp_rmse: float) -> float:
    """Positive => PP worse than direct on the held-out unit."""
    return (pp_rmse - direct_rmse) / max(direct_rmse, 1e-12)


def loo_for_dataset(saved, gain, win, worst):
    y = np.asarray(saved["validation_y"], float)
    groups = np.asarray(saved["validation_groups"])
    direct = np.asarray(saved["validation_plain"], float).mean(0)
    pp = np.asarray(saved["validation_pp"], float).mean(0)
    units = list(np.unique(groups))
    if len(units) < MIN_VAL_UNITS:
        return None

    full = evidence_from_units(y, groups, direct, pp, units)
    full_accept = accepts_pp(full, gain, win, worst)
    direct_u = unit_rmse(y, groups, direct)
    pp_u = unit_rmse(y, groups, pp)

    folds = []
    for held in units:
        keep = [u for u in units if u != held]
        ev = evidence_from_units(y, groups, direct, pp, keep)
        if ev is None or ev["n_units"] < 1:
            continue
        accept = accepts_pp(ev, gain, win, worst)
        regret = relative_regret(direct_u[held], pp_u[held])
        # Policy regret: if reject PP, deploy direct → regret 0 on this metric.
        policy_regret = regret if accept else 0.0
        folds.append(
            {
                "held_out": str(held),
                "accept_pp": bool(accept),
                "matches_full": bool(accept == full_accept),
                "unit_helped": bool(regret < 0),
                "raw_pp_regret": float(regret),
                "policy_regret": float(policy_regret),
            }
        )

    if not folds:
        return None

    accepted_folds = [f for f in folds if f["accept_pp"]]
    win_frac = (
        float(np.mean([f["unit_helped"] for f in accepted_folds]))
        if accepted_folds
        else None
    )
    policy_regrets = np.asarray([f["policy_regret"] for f in folds], float)
    return {
        "n_units": len(units),
        "n_folds": len(folds),
        "full_accept_pp": bool(full_accept),
        "oof_accept_rate": float(np.mean([f["accept_pp"] for f in folds])),
        "route_stability": float(np.mean([f["matches_full"] for f in folds])),
        "oof_executor_win_fraction": win_frac,
        "oof_mean_policy_regret": float(np.mean(policy_regrets)),
        "oof_worst_policy_regret": float(np.max(policy_regrets)),
        "n_accept_folds": len(accepted_folds),
    }


def aggregate(cells_by_dataset: dict) -> dict:
    """Macro-average OOF metrics across eligible datasets."""
    rows = [row for row in cells_by_dataset.values() if row is not None]
    if not rows:
        return {}
    win_vals = [
        row["oof_executor_win_fraction"]
        for row in rows
        if row["oof_executor_win_fraction"] is not None
    ]
    return {
        "n_datasets": len(rows),
        "macro_route_stability": float(np.mean([r["route_stability"] for r in rows])),
        "macro_oof_accept_rate": float(np.mean([r["oof_accept_rate"] for r in rows])),
        "macro_oof_executor_win_fraction": (
            float(np.mean(win_vals)) if win_vals else None
        ),
        "macro_oof_mean_policy_regret": float(
            np.mean([r["oof_mean_policy_regret"] for r in rows])
        ),
        "macro_oof_worst_policy_regret": float(
            np.mean([r["oof_worst_policy_regret"] for r in rows])
        ),
        "max_dataset_worst_policy_regret": float(
            np.max([r["oof_worst_policy_regret"] for r in rows])
        ),
        "datasets_full_accept": int(sum(r["full_accept_pp"] for r in rows)),
    }


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    names = (
        list(metadata["datasets"].keys())
        if isinstance(metadata["datasets"], dict)
        else list(metadata["datasets"])
    )

    eligibility = {}
    for name in names:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        n_units = len(np.unique(saved["validation_groups"]))
        eligibility[name] = {
            "validation_units": int(n_units),
            "eligible_for_loo": n_units >= MIN_VAL_UNITS,
        }

    eligible = [n for n, row in eligibility.items() if row["eligible_for_loo"]]
    skipped = [n for n, row in eligibility.items() if not row["eligible_for_loo"]]

    grid = []
    for gain, win, worst in product(GAINS, WINS, WORSTS):
        per_dataset = {}
        for name in eligible:
            saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
            per_dataset[name] = loo_for_dataset(saved, gain, win, worst)
        summary = aggregate(per_dataset)
        grid.append(
            {
                "gain": gain,
                "unit_win": win,
                "worst_ratio": worst,
                "is_frozen_default": (gain, win, worst) == FROZEN,
                **summary,
                "per_dataset": per_dataset,
            }
        )

    # Rank by robustness: high route stability, low worst regret, then high win frac.
    # Do NOT rank by test accuracy.
    def robustness_key(row):
        return (
            -row["macro_route_stability"],
            row["max_dataset_worst_policy_regret"],
            row["macro_oof_mean_policy_regret"],
            -(row["macro_oof_executor_win_fraction"] or 0.0),
            row["gain"],
            row["unit_win"],
            row["worst_ratio"],
        )

    ranked = sorted(grid, key=robustness_key)
    frozen = next(row for row in grid if row["is_frozen_default"])
    frozen_rank = next(
        i for i, row in enumerate(ranked) if row["is_frozen_default"]
    ) + 1

    # Robust region: stability ≥ frozen and worst regret ≤ frozen.
    robust_region = [
        row
        for row in grid
        if row["macro_route_stability"] >= frozen["macro_route_stability"] - 1e-12
        and row["max_dataset_worst_policy_regret"]
        <= frozen["max_dataset_worst_policy_regret"] + 1e-12
    ]

    payload = {
        "status": (
            "OOF threshold-policy robustness audit on validation units; "
            "test labels unused for threshold choice"
        ),
        "owner": "박진서",
        "philosophy": (
            "Among a pre-declared operational grid, prefer policies whose "
            "unit-holdout approval decisions stay consistent and whose "
            "worst-case policy regret stays small. Do not pick τ by test SOTA."
        ),
        "protocol": {
            "gains": list(GAINS),
            "unit_wins": list(WINS),
            "worst_ratios": list(WORSTS),
            "loo": "leave-one-validation-unit-out",
            "min_validation_units": MIN_VAL_UNITS,
            "eligible_datasets": eligible,
            "skipped_datasets": skipped,
            "eligibility": eligibility,
            "frozen_target": {
                "gain": FROZEN[0],
                "unit_win": FROZEN[1],
                "worst_ratio": FROZEN[2],
            },
        },
        "frozen_cell": {
            k: frozen[k]
            for k in frozen
            if k != "per_dataset"
        },
        "frozen_robustness_rank": frozen_rank,
        "robust_region_size": len(robust_region),
        "ranking_by_oof_robustness": [
            {
                "rank": i + 1,
                "gain": row["gain"],
                "unit_win": row["unit_win"],
                "worst_ratio": row["worst_ratio"],
                "is_frozen_default": row["is_frozen_default"],
                "macro_route_stability": row["macro_route_stability"],
                "macro_oof_executor_win_fraction": row[
                    "macro_oof_executor_win_fraction"
                ],
                "macro_oof_mean_policy_regret": row["macro_oof_mean_policy_regret"],
                "max_dataset_worst_policy_regret": row[
                    "max_dataset_worst_policy_regret"
                ],
                "macro_oof_accept_rate": row["macro_oof_accept_rate"],
            }
            for i, row in enumerate(ranked)
        ],
        "grid": [
            {k: v for k, v in row.items() if k != "per_dataset"} for row in grid
        ],
        "per_dataset_frozen": frozen["per_dataset"],
        "deployment_story": {
            "1": "Development units → OOF robustness over declared τ grid",
            "2": "Freeze τ=(2%, 60%, 1.10) as an operational point inside the stable region",
            "3": "New dataset: do not retune τ",
            "4": "That dataset's validation approves/rejects executors under frozen τ",
            "5": "Test remains untouched",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print("eligible", eligible)
    print("skipped", skipped)
    print(
        "frozen",
        {
            k: frozen[k]
            for k in (
                "macro_route_stability",
                "macro_oof_executor_win_fraction",
                "macro_oof_mean_policy_regret",
                "max_dataset_worst_policy_regret",
            )
        },
        "rank",
        frozen_rank,
        "/",
        len(grid),
    )
    print("top3", payload["ranking_by_oof_robustness"][:3])


if __name__ == "__main__":
    main()
