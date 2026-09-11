#!/usr/bin/env python3
"""Noskov et al. Algorithm-1 audit at PP-X-matched coverage."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from final_modular_pp_evidence import build_datasets, exact_sign_flip  # noqa: E402
from full_equal_candidate_budget import SETTINGS, load_settings  # noqa: E402

OUT = ROOT / "results/selective_regression_equal_coverage_v1"
NAME = {
    "hust": "HUST",
    "virkler": "Virkler",
    "nasa": "NASA",
    "sunwoda": "SUNWODA",
    "rwth": "RWTH",
    "mich": "MICH",
    "matr": "MATR2019",
    "matr_batch2": "MATR-b2",
    "ncmapss": "N-CMAPSS",
}
COVERAGES = (0.25, 0.50, 0.75, 0.90)
BETA = 0.10
CHUNK = 256


def flatten(parts):
    if isinstance(parts, tuple):
        return parts
    result = []
    for split in range(3):
        result.append({
            key: np.concatenate([fold[split][key] for fold in parts])
            for key in ("x", "y", "groups")
        })
    return tuple(result)


def representation(train_x, validation_x, test_x):
    train_x = np.asarray(train_x, float)
    center = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0

    def clean(x):
        z = (np.asarray(x, float) - center) / scale
        return np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0)

    tr, va, te = map(clean, (train_x, validation_x, test_x))
    d = max(1, min(3, tr.shape[1], len(tr) - 1))
    pca = PCA(n_components=d, random_state=42).fit(tr)
    return pca.transform(tr), pca.transform(va), pca.transform(te)


def median_nn_distance(x):
    nn = NearestNeighbors(n_neighbors=2).fit(x)
    distance = nn.kneighbors(x, return_distance=True)[0][:, 1]
    positive = distance[np.isfinite(distance) & (distance > 1e-12)]
    return float(np.median(positive)) if len(positive) else 1.0


def kernel_estimates(train_x, train_y, query_x, h):
    """Gaussian NW mean, variance, density, and Algorithm-1 score."""
    n, d = train_x.shape
    normalizer = (2 * math.pi) ** (-d / 2)
    means, variances, densities, brackets = [], [], [], []
    z = float(ndtri(1 - BETA))
    kernel_l2_sq = 2 ** (-d) * math.pi ** (-d / 2)
    for start in range(0, len(query_x), CHUNK):
        q = query_x[start:start + CHUNK]
        sq = np.sum((q[:, None, :] - train_x[None, :, :]) ** 2, axis=2)
        kernel = normalizer * np.exp(-0.5 * sq / max(h * h, 1e-16))
        sums = kernel.sum(axis=1)
        weights = kernel / np.maximum(sums[:, None], 1e-300)
        mean = weights @ train_y
        second = weights @ np.square(train_y)
        variance = np.maximum(second - np.square(mean), 0.0)
        density = sums / (n * h ** d)
        correction = z * math.sqrt(2 * kernel_l2_sq) / np.sqrt(
            np.maximum(n * h ** d * density, 1e-300)
        )
        means.append(mean)
        variances.append(variance)
        densities.append(density)
        brackets.append(1 - correction)
    return tuple(np.concatenate(v) for v in (means, variances, densities, brackets))


def algorithm_score(variance, density, bracket, n, d, h):
    a = (2 * math.pi) ** (-d / 2) * math.exp(-0.5)
    density_ok = density >= 4 * a / (n * h ** d)
    valid = density_ok & (bracket > 0)
    score = np.full(len(variance), np.inf)
    score[valid] = variance[valid] / bracket[valid]
    return score, density_ok


def choose(indices_score, k):
    order = np.lexsort((np.arange(len(indices_score)), indices_score))
    finite = order[np.isfinite(indices_score[order])]
    return finite[: min(k, len(finite))]


def metrics(y, prediction, groups, selected):
    if len(selected) == 0:
        return {"n": 0, "normalized_rmse": None, "unit_macro_normalized_rmse": None,
                "unit_coverage": 0.0}
    scale = max(float(np.std(y)), 1e-12)
    rmse = math.sqrt(float(np.mean((y[selected] - prediction[selected]) ** 2)))
    unit_values = []
    selected_mask = np.zeros(len(y), bool)
    selected_mask[selected] = True
    for group in np.unique(groups):
        mask = (groups == group) & selected_mask
        if np.any(mask):
            unit_values.append(math.sqrt(float(np.mean((y[mask] - prediction[mask]) ** 2))) / scale)
    return {
        "n": int(len(selected)),
        "normalized_rmse": rmse / scale,
        "unit_macro_normalized_rmse": float(np.mean(unit_values)),
        "unit_coverage": len(unit_values) / len(np.unique(groups)),
    }


def main():
    loaded = load_settings(SETTINGS)
    frozen = build_datasets()
    rows = []
    for setting in SETTINGS:
        parts = flatten(loaded[setting])
        train, validation, test = parts
        tr, va, te = representation(train["x"], validation["x"], test["x"])
        base_h = median_nn_distance(tr)
        candidates = [max(base_h * multiplier, 1e-4) for multiplier in (0.5, 1, 2, 4)]
        search = []
        for h in candidates:
            mean, variance, density, bracket = kernel_estimates(tr, train["y"], va, h)
            clipped = np.clip(mean, 0, max(float(np.max(train["y"])), 0))
            search.append({"h": h, "validation_mse": float(np.mean((validation["y"] - clipped) ** 2))})
        selected_h = min(search, key=lambda item: (item["validation_mse"], item["h"]))["h"]
        nw, variance, density, bracket = kernel_estimates(tr, train["y"], te, selected_h)
        nw = np.clip(nw, 0, max(float(np.max(train["y"])), 0))
        score, density_ok = algorithm_score(
            variance, density, bracket, len(tr), tr.shape[1], selected_h
        )

        y, groups, pp_seeds = frozen[NAME[setting]][:3]
        y, groups, pp_seeds = np.asarray(y), np.asarray(groups).astype(str), np.asarray(pp_seeds)
        assert np.allclose(y, test["y"]), f"target mismatch: {setting}"
        assert np.array_equal(groups, np.asarray(test["groups"]).astype(str)), f"group mismatch: {setting}"
        pp = pp_seeds.mean(axis=0)
        pp_score = pp_seeds.std(axis=0) / max(float(np.std(validation["y"])), 1e-12)

        coverage_results = {}
        for coverage in COVERAGES:
            target_k = max(1, int(math.floor(coverage * len(y))))
            nw_selected = choose(score, target_k)
            achieved_k = len(nw_selected)
            pp_selected = np.lexsort((np.arange(len(pp_score)), pp_score))[:achieved_k]
            score_selected = choose(score, achieved_k)
            coverage_results[str(coverage)] = {
                "target_rows": target_k,
                "achieved_rows": achieved_k,
                "achieved_coverage": achieved_k / len(y),
                "selective_nw": metrics(y, nw, groups, nw_selected),
                "ppx_seed_score": metrics(y, pp, groups, pp_selected),
                "ppx_algorithm1_score": metrics(y, pp, groups, score_selected),
            }
        rows.append({
            "dataset": NAME[setting],
            "n_train": len(train["y"]),
            "n_validation": len(validation["y"]),
            "n_test": len(test["y"]),
            "pca_dimension": tr.shape[1],
            "bandwidth_search": search,
            "selected_bandwidth": selected_h,
            "algorithm1_density_admissible_coverage": float(np.mean(density_ok & (bracket > 0))),
            "coverage": coverage_results,
        })
        print(setting, "admissible", rows[-1]["algorithm1_density_admissible_coverage"])

    aggregate = {}
    for coverage in COVERAGES:
        key = str(coverage)
        usable = [row for row in rows if row["coverage"][key]["achieved_rows"] > 0]
        ppx = np.asarray([row["coverage"][key]["ppx_seed_score"]["normalized_rmse"] for row in usable])
        selective = np.asarray([row["coverage"][key]["selective_nw"]["normalized_rmse"] for row in usable])
        effects = selective - ppx
        aggregate[key] = {
            "usable_datasets": len(usable),
            "mean_achieved_coverage": float(np.mean([
                row["coverage"][key]["achieved_coverage"] for row in rows
            ])),
            "ppx_macro_normalized_rmse": float(np.mean(ppx)) if len(ppx) else None,
            "selective_nw_macro_normalized_rmse": float(np.mean(selective)) if len(selective) else None,
            "ppx_dataset_wins": int(np.sum(effects > 0)),
            "paired_exact_sign_flip_p": exact_sign_flip(effects) if len(effects) else None,
            "normalized_rmse_difference_selective_minus_ppx": effects.tolist(),
        }
    payload = {
        "protocol": "protocols/SELECTIVE_REGRESSION_EQUAL_COVERAGE_PROTOCOL.md",
        "paper": "Noskov, Fishkov, Panov, Selective Nonparametric Regression via Testing, Algorithm 1",
        "beta": BETA,
        "coverages": list(COVERAGES),
        "datasets": rows,
        "aggregate": aggregate,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
