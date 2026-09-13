#!/usr/bin/env python3
"""Threshold sensitivity of frozen PP-X approval criteria.

Grid
----
gain ∈ {1%, 2%, 5%}
unit-win ∈ {50%, 60%, 70%}
worst-ratio ∈ {1.05, 1.10, 1.20}

Uses the frozen 12-setting common-backbone prediction archive. Decisions use
validation evidence only; test labels score false accept / reject after the
fact. CI gate is intentionally off so the audit matches the operational
thresholds cited in the paper (2% / 60% / 1.10), not the stricter CI variant.
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
from ppx_paper_policy_audit import test_effect, validation_evidence  # noqa: E402

SOURCE = ROOT / "results/cross_domain_mechanism_v2"
OUT = ROOT / "results/ppx_threshold_sensitivity_v1"

GAINS = (0.01, 0.02, 0.05)
WINS = (0.50, 0.60, 0.70)
WORSTS = (1.05, 1.10, 1.20)
FROZEN = {"gain": 0.02, "win": 0.60, "worst": 1.10}


def decide(validation: dict, *, gain: float, win: float, worst: float) -> bool:
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
                unit_gain_ci_low=validation.get("unit_gain_ci_low", float("-inf")),
            ),
        ),
        min_relative_improvement=gain,
        min_unit_win_fraction=win,
        max_worst_unit_rmse_ratio=worst,
        min_unit_gain_ci_low=float("-inf"),
    )
    return decision.executor == "unbounded"


def score(selected: np.ndarray, actual: np.ndarray, effects: np.ndarray) -> dict:
    chosen_effects = np.where(selected, effects, 0.0)
    return {
        "accepted": int(selected.sum()),
        "correct": int(np.sum(selected == actual)),
        "accuracy": float(np.mean(selected == actual)),
        "false_accepts": int(np.sum(selected & ~actual)),
        "false_rejects": int(np.sum(~selected & actual)),
        "equal_dataset_mean_log_rmse_ratio": float(np.mean(chosen_effects)),
    }


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    validations = []
    actual = []
    effects = []
    names = []
    for name in metadata["datasets"]:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        validation = validation_evidence(saved)
        test = test_effect(saved)
        names.append(name)
        validations.append(validation)
        actual.append(bool(test["pp_helped"]))
        effects.append(float(test["mean_unit_log_rmse_ratio"]))
    actual_arr = np.asarray(actual, dtype=bool)
    effects_arr = np.asarray(effects, float)

    # Per-setting validation snapshot for the appendix.
    snapshots = []
    for name, validation, helped in zip(names, validations, actual):
        snapshots.append(
            {
                "dataset": name,
                "validation_relative_gain": 1.0
                - validation["pp_loss"] / max(validation["direct_loss"], 1e-12),
                "unit_win_fraction": validation["unit_win_fraction"],
                "worst_unit_ratio": validation["worst_unit_ratio"],
                "test_pp_helped": helped,
            }
        )

    grid = []
    for gain, win, worst in product(GAINS, WINS, WORSTS):
        selected = np.asarray(
            [
                decide(validation, gain=gain, win=win, worst=worst)
                for validation in validations
            ],
            dtype=bool,
        )
        summary = score(selected, actual_arr, effects_arr)
        accepted_names = [name for name, flag in zip(names, selected) if flag]
        grid.append(
            {
                "gain": gain,
                "unit_win": win,
                "worst_ratio": worst,
                "is_frozen_default": (
                    gain == FROZEN["gain"]
                    and win == FROZEN["win"]
                    and worst == FROZEN["worst"]
                ),
                **summary,
                "accepted_datasets": accepted_names,
            }
        )

    frozen = next(row for row in grid if row["is_frozen_default"])
    simple = np.asarray(
        [v["pp_loss"] < v["direct_loss"] for v in validations], dtype=bool
    )
    simple_summary = score(simple, actual_arr, effects_arr)

    # One-factor slices around the frozen point.
    slices = {"gain": [], "unit_win": [], "worst_ratio": []}
    for row in grid:
        if row["unit_win"] == FROZEN["win"] and row["worst_ratio"] == FROZEN["worst"]:
            slices["gain"].append(row)
        if row["gain"] == FROZEN["gain"] and row["worst_ratio"] == FROZEN["worst"]:
            slices["unit_win"].append(row)
        if row["gain"] == FROZEN["gain"] and row["unit_win"] == FROZEN["win"]:
            slices["worst_ratio"].append(row)

    fa_values = [row["false_accepts"] for row in grid]
    acc_values = [row["correct"] for row in grid]
    payload = {
        "status": "retrospective threshold sensitivity on 12 common-backbone settings",
        "owner": "박진서",
        "protocol": {
            "gains": list(GAINS),
            "unit_wins": list(WINS),
            "worst_ratios": list(WORSTS),
            "ci_gate": "off (matches operational 2%/60%/1.10 claim)",
            "n_settings": len(names),
            "source": str(SOURCE.relative_to(ROOT)),
        },
        "baselines": {
            "simple_validation_rmse_only": simple_summary,
            "frozen_2_60_1p10": {
                k: frozen[k]
                for k in (
                    "accepted",
                    "correct",
                    "accuracy",
                    "false_accepts",
                    "false_rejects",
                    "equal_dataset_mean_log_rmse_ratio",
                    "accepted_datasets",
                )
            },
        },
        "grid": grid,
        "slices_around_frozen": slices,
        "stability": {
            "grid_size": len(grid),
            "false_accepts_range": [min(fa_values), max(fa_values)],
            "correct_range": [min(acc_values), max(acc_values)],
            "cells_with_fa_le_frozen": int(
                sum(row["false_accepts"] <= frozen["false_accepts"] for row in grid)
            ),
            "cells_with_correct_ge_frozen": int(
                sum(row["correct"] >= frozen["correct"] for row in grid)
            ),
            "cells_with_fa_le_2_and_correct_ge_8": int(
                sum(
                    row["false_accepts"] <= 2 and row["correct"] >= 8
                    for row in grid
                )
            ),
            "cells_matching_frozen_accept_set": int(
                sum(
                    row["accepted_datasets"] == frozen["accepted_datasets"]
                    for row in grid
                )
            ),
        },
        "dataset_snapshots": snapshots,
        "interpretation": (
            "Thresholds are operational, not theoretically optimal. Sensitivity "
            "asks whether nearby values preserve false-accept control versus "
            "RMSE-only selection."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload["baselines"], indent=2))
    print("stability", json.dumps(payload["stability"], indent=2))
    print("frozen", frozen["accepted_datasets"], "FA", frozen["false_accepts"], "correct", frozen["correct"])


if __name__ == "__main__":
    main()
