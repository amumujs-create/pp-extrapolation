#!/usr/bin/env python3
"""Outcome-blind-at-test PP/MLP route audit on matched frozen splits.

The gate is deliberately parameter-free: select PP only when its five-seed
ensemble has lower validation RMSE than the matched MLP ensemble.  The test
labels are opened only after the route is determined.  This is retrospective
because the split collection and model family were developed previously.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "cross_domain_mechanism_v2"
OUTPUT = ROOT / "results" / "journal_validation_route_v1"
RNG = np.random.default_rng(20260908)


def unit_relative_gain(y: np.ndarray, g: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    values = []
    for unit in np.unique(g):
        mask = g == unit
        a = np.sqrt(np.mean((y[mask] - base[mask]) ** 2))
        b = np.sqrt(np.mean((y[mask] - candidate[mask]) ** 2))
        values.append((a - b) / max(a, 1e-10))
    return np.asarray(values)


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    rows = []
    for name in metadata["datasets"]:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        validation_y = np.asarray(saved["validation_y"], float)
        validation_plain = saved["validation_plain"].mean(0)
        validation_pp = saved["validation_pp"].mean(0)
        plain_rmse = float(np.sqrt(np.mean((validation_y - validation_plain) ** 2)))
        pp_rmse = float(np.sqrt(np.mean((validation_y - validation_pp) ** 2)))
        selected = "PP" if pp_rmse < plain_rmse else "MLP"

        # The route is fixed above before these arrays are used for scoring.
        y = np.asarray(saved["y"], float)
        groups = np.asarray(saved["groups"])
        plain = saved["plain"].mean(0)
        pp = saved["pp"].mean(0)
        chosen = pp if selected == "PP" else plain
        delta_pp = unit_relative_gain(y, groups, plain, pp)
        delta_route = unit_relative_gain(y, groups, plain, chosen)
        rows.append({
            "dataset": name,
            "validation_plain_rmse": plain_rmse,
            "validation_pp_rmse": pp_rmse,
            "validation_relative_gain": (plain_rmse - pp_rmse) / max(plain_rmse, 1e-10),
            "selected": selected,
            "test_pp_mean_relative_unit_gain": float(delta_pp.mean()),
            "test_route_mean_relative_unit_gain": float(delta_route.mean()),
            "test_pp_helped": bool(delta_pp.mean() > 0),
            "selection_correct": bool((selected == "PP") == (delta_pp.mean() > 0)),
            "n_units": len(delta_pp),
            "pp_unit_wins": int(np.sum(delta_pp > 0)),
        })

    actual = np.asarray([row["test_pp_helped"] for row in rows], int)
    predicted = np.asarray([row["selected"] == "PP" for row in rows], int)
    recalls = [float(np.mean(predicted[actual == value] == value)) for value in (0, 1) if np.any(actual == value)]
    correct = int(np.sum(actual == predicted))
    route_gain = np.asarray([row["test_route_mean_relative_unit_gain"] for row in rows])
    pp_gain = np.asarray([row["test_pp_mean_relative_unit_gain"] for row in rows])

    # Equal-dataset bootstrap. This is an uncertainty interval over this fixed
    # benchmark collection, not a population claim about all future domains.
    index = RNG.integers(0, len(rows), size=(50000, len(rows)))
    route_boot = route_gain[index].mean(1)
    pp_boot = pp_gain[index].mean(1)
    result = {
        "experiment": "journal_validation_route_v1",
        "status": "retrospective, parameter-free validation-only route audit",
        "gate": "select PP iff five-seed validation ensemble RMSE is lower than matched MLP",
        "n_datasets": len(rows),
        "selection": {
            "correct": correct,
            "accuracy": correct / len(rows),
            "balanced_accuracy": float(np.mean(recalls)),
            "false_pp_accepts": int(np.sum((predicted == 1) & (actual == 0))),
            "false_pp_rejects": int(np.sum((predicted == 0) & (actual == 1))),
            "two_sided_sign_test_p_vs_half": float(binomtest(correct, len(rows), 0.5).pvalue),
        },
        "always_pp": {
            "equal_dataset_mean_relative_unit_gain": float(pp_gain.mean()),
            "bootstrap_ci95": np.quantile(pp_boot, [0.025, 0.975]).tolist(),
            "dataset_wins": int(np.sum(pp_gain > 0)),
        },
        "validation_route": {
            "equal_dataset_mean_relative_unit_gain": float(route_gain.mean()),
            "bootstrap_ci95": np.quantile(route_boot, [0.025, 0.975]).tolist(),
            "dataset_improvements_over_plain": int(np.sum(route_gain > 0)),
            "dataset_losses_vs_plain": int(np.sum(route_gain < 0)),
        },
        "datasets": rows,
        "warning": "The zero-loss MLP fallback is structural: rejected PP routes exactly equal MLP. This tests model routing, not abstention or a prospective external gate.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
