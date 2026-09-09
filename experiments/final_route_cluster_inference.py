#!/usr/bin/env python3
"""Inference only where final PP-X and 30-candidate baselines share exact test rows."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "results/final_30_candidate_competitors_v1/predictions"
OUT = ROOT / "results/final_route_cluster_inference_v1"

# Dataset, final-route artifact, target/prediction/group keys.  Every entry is
# checked for bytewise-identical y and group vectors before inference.
ROUTES = {
    "hust": ("results/hust_regime_transport_pp_v1/predictions.npz", "y", "transported", "groups"),
    "xjtu": ("results/xjtu_reflected_scale_pp_v3/predictions.npz", "y", "predictions", "groups"),
    "matr": ("results/matr_pp_validation_calibration_v1/predictions.npz", "y", "prediction", "groups"),
    "matr_batch2": ("results/matr_batch2_pp_five_seed_replay/predictions.npz", "truth", "transported", "groups"),
    "milling": ("results/milling_boundary_quotient_route_v1/predictions.npz", "y", "prediction", "groups"),
}


def r2(y, p):
    denom = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / denom) if denom else float("nan")


def bootstrap(y, ppx, base, groups, rng, repeats=20000):
    units = np.unique(groups); draws = []
    unit_indices = [np.flatnonzero(groups == u) for u in units]
    for i in range(repeats):
        ind = np.concatenate([unit_indices[j] for j in rng.integers(len(units), size=len(units))])
        value = r2(y[ind], ppx[ind]) - r2(y[ind], base[ind])
        if np.isfinite(value):
            draws.append(value)
    # A small tail may yield zero target variance after resampling.  Do not
    # turn its conditional finite subset into an apparently valid CI.
    if len(draws) < .99 * repeats:
        return None
    return [float(x) for x in np.quantile(draws, (.025, .5, .975))]


def sign_flip(y, ppx, base, groups, rng):
    units = np.unique(groups)
    gains = np.array([np.mean((y[groups == u] - base[groups == u]) ** 2) -
                      np.mean((y[groups == u] - ppx[groups == u]) ** 2) for u in units])
    observed = abs(float(gains.mean()))
    if len(units) <= 20:
        signs = np.array([[1 if b & (1 << j) else -1 for j in range(len(units))]
                          for b in range(1 << len(units))])
        p = np.mean(np.abs(signs @ gains / len(units)) >= observed - 1e-12)
        method = "exact"
    else:
        # Labelled Monte-Carlo permutation; fixed RNG makes it reproducible.
        signs = rng.choice((-1, 1), size=(100000, len(units)))
        p = np.mean(np.abs(signs @ gains / len(units)) >= observed - 1e-12)
        method = "monte_carlo_100000"
    return float(p), float(gains.mean()), method


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"protocol": "final PP-X route vs 30-candidate/5-seed baseline; exact row audit; unit-cluster bootstrap (20k); paired sign-flip", "datasets": {}}
    for name, (path, ykey, pkey, gkey) in ROUTES.items():
        z = np.load(ROOT / path, allow_pickle=True)
        y, groups, ppx = z[ykey], z[gkey], z[pkey].mean(axis=0)
        entry = {"n_units": int(len(np.unique(groups))), "ppx_r2": r2(y, ppx), "comparisons": {}}
        for baseline in ("vrex", "groupdro", "monotone"):
            b = np.load(COMP / f"{name}_{baseline}.npz", allow_pickle=True)
            if not (np.array_equal(y, b["y"]) and np.array_equal(groups.astype(str), b["groups"].astype(str))):
                raise RuntimeError(f"{name}/{baseline}: test row mismatch")
            pred = b["prediction"].mean(axis=0)
            pvalue, gain, method = sign_flip(y, ppx, pred, groups, np.random.default_rng(20260910))
            entry["comparisons"][baseline] = {
                "baseline_r2": r2(y, pred), "delta_r2": r2(y, ppx) - r2(y, pred),
                "delta_r2_cluster_bootstrap_95": bootstrap(y, ppx, pred, groups, np.random.default_rng(20260910)),
                "unit_mse_gain_ppx_minus_baseline": gain, "sign_flip": method, "paired_sign_flip_pvalue": pvalue,
            }
        out["datasets"][name] = entry
    (OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
