#!/usr/bin/env python3
"""Cluster bootstrap and paired permutation inference for final MICH predictions."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PPX = ROOT.parent / "ca-css-ncmapss/results/final_ppx_mich_matched_v1/predictions.npz"
COMP = ROOT / "results/final_30_candidate_competitors_v1/predictions"
OUT = ROOT / "results/mich_cluster_inference_v1"


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def bootstrap(y, ppx, baseline, groups, rng, repeats=20000):
    units = np.unique(groups); draws = []
    for _ in range(repeats):
        chosen = rng.choice(units, len(units), replace=True)
        index = np.concatenate([np.flatnonzero(groups == unit) for unit in chosen])
        draws.append(r2(y[index], ppx[index]) - r2(y[index], baseline[index]))
    return [float(x) for x in np.quantile(draws, (.025, .5, .975))]


def exact_sign_flip(y, ppx, baseline, groups):
    units = np.unique(groups)
    loss_delta = np.asarray([np.mean((y[groups == unit] - baseline[groups == unit]) ** 2) -
                             np.mean((y[groups == unit] - ppx[groups == unit]) ** 2) for unit in units])
    observed = abs(float(loss_delta.mean())); exceed = 0
    for bits in range(1 << len(units)):
        signs = np.array([1 if bits & (1 << j) else -1 for j in range(len(units))])
        exceed += abs(float(np.mean(signs * loss_delta))) >= observed - 1e-12
    return float(exceed / (1 << len(units))), float(loss_delta.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(PPX); y, groups, ppx = z["truth"], z["groups"], z["final_ppx_dual_scale"]
    result = {"protocol": "unit-cluster bootstrap (20k) and exact paired unit sign-flip on seed-average predictions",
              "n_units": int(len(np.unique(groups))), "comparisons": {}}
    for name in ("vrex", "groupdro", "monotone"):
        c = np.load(COMP / f"mich_{name}.npz")
        if not (np.array_equal(y, c["y"]) and np.array_equal(groups, c["groups"])):
            raise RuntimeError(f"row mismatch for {name}")
        pred = c["prediction"].mean(axis=0)
        pvalue, mean_gain = exact_sign_flip(y, ppx, pred, groups)
        result["comparisons"][name] = {"ppx_r2": r2(y, ppx), "baseline_r2": r2(y, pred),
            "delta_r2": r2(y, ppx) - r2(y, pred),
            "delta_r2_cluster_bootstrap_95": bootstrap(y, ppx, pred, groups, np.random.default_rng(20260910)),
            "unit_mse_gain_ppx_minus_baseline": mean_gain, "exact_sign_flip_pvalue": pvalue}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
