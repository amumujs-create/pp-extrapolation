#!/usr/bin/env python3
"""Aggregate the completed 30-candidate baseline matrix against frozen PP-X."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from final_modular_pp_evidence import (  # noqa: E402
    bh_adjust,
    bootstrap_ci,
    build_datasets,
    exact_sign_flip,
)

SOURCE = ROOT / "results/full_equal_candidate_budget_v1"
OUT = ROOT / "results/full_equal_candidate_budget_summary_v1"
MODELS = (
    "plain_mlp",
    "ft_transformer",
    "vrex",
    "groupdro",
    "monotone",
    "engression",
    "linear_tail_rbf",
    "svgp",
)
NAMES = {
    "HUST": "hust",
    "Virkler": "virkler",
    "NASA": "nasa",
    "SUNWODA": "sunwoda",
    "RWTH": "rwth",
    "MICH": "mich",
    "MATR2019": "matr",
    "MATR-b2": "matr_batch2",
    "N-CMAPSS": "ncmapss",
}
RNG_SEED = 20260911


def r2(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def baseline(setting: str, model: str):
    if setting != "nasa":
        z = np.load(
            SOURCE / setting / "main" / model / "predictions.npz",
            allow_pickle=True,
        )
        return z["y"], z["groups"].astype(str), z["prediction"]
    ys, groups, predictions = [], [], []
    for fold in range(4):
        z = np.load(
            SOURCE / setting / f"fold_{fold}" / model / "predictions.npz",
            allow_pickle=True,
        )
        ys.append(z["y"])
        groups.append(z["groups"].astype(str))
        predictions.append(z["prediction"])
    return np.concatenate(ys), np.concatenate(groups), np.concatenate(predictions, axis=1)


def main():
    frozen = build_datasets()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for paper_name, values in frozen.items():
        y, groups, pp = map(np.asarray, values[:3])
        groups = groups.astype(str)
        scores, artifacts = {}, {}
        for model in MODELS:
            by, bg, prediction = baseline(NAMES[paper_name], model)
            assert np.allclose(y, by), f"target mismatch {paper_name}/{model}"
            assert np.array_equal(groups, bg), f"group mismatch {paper_name}/{model}"
            assert prediction.shape == pp.shape == (5, len(y))
            scores[model] = r2(y, prediction.mean(0))
            artifacts[model] = prediction
        strongest = max(scores, key=scores.get)
        comparator = artifacts[strongest].mean(0)
        selected = pp.mean(0)
        effects = []
        for unit in np.unique(groups):
            mask = groups == unit
            pp_rmse = math.sqrt(float(np.mean((y[mask] - selected[mask]) ** 2)))
            base_rmse = math.sqrt(float(np.mean((y[mask] - comparator[mask]) ** 2)))
            effects.append(math.log(max(base_rmse, 1e-12) / max(pp_rmse, 1e-12)))
        effects = np.asarray(effects)
        rows.append({
            "dataset": paper_name,
            "ppx_r2": r2(y, selected),
            "baseline_r2": scores,
            "strongest_equal_budget_baseline": strongest,
            "strongest_equal_budget_r2": scores[strongest],
            "r2_gap": r2(y, selected) - scores[strongest],
            "n_units": len(effects),
            "units_won": int(np.sum(effects > 0)),
            "mean_unit_log_rmse_ratio": float(effects.mean()),
            "unit_bootstrap_ci95": bootstrap_ci(effects, rng),
            "unit_sign_flip_p": exact_sign_flip(effects),
            "unit_log_rmse_ratios": effects.tolist(),
        })
    q = bh_adjust([row["unit_sign_flip_p"] for row in rows])
    for row, value in zip(rows, q):
        row["unit_bh_q"] = value
    differences = np.asarray([row["r2_gap"] for row in rows])
    positive = int(np.sum(differences > 0))
    dataset_sign_p = float(
        2 * sum(math.comb(len(rows), k) for k in range(positive, len(rows) + 1))
        / 2 ** len(rows)
    )
    all_effects = np.concatenate([np.asarray(row["unit_log_rmse_ratios"]) for row in rows])
    payload = {
        "protocol": (
            "eight baselines; exactly 30 validation candidates each; search seed 42; "
            "selected refit seeds 42--46; frozen PP-X is not counted as equal-budget search"
        ),
        "warning": (
            "The strongest baseline is selected descriptively using test R2. Paired effects "
            "against that winner are conservative secondary evidence, not a prespecified test."
        ),
        "datasets": rows,
        "aggregate": {
            "ppx_pooled_wins": positive,
            "datasets": len(rows),
            "dataset_sign_test_two_sided": dataset_sign_p,
            "total_physical_units": int(sum(row["n_units"] for row in rows)),
            "unit_weighted_mean_log_rmse_ratio": float(all_effects.mean()),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(
            row["dataset"],
            f"PP-X={row['ppx_r2']:.3f}",
            f"{row['strongest_equal_budget_baseline']}={row['strongest_equal_budget_r2']:.3f}",
            f"gap={row['r2_gap']:+.3f}",
        )
    print(payload["aggregate"])


if __name__ == "__main__":
    main()
