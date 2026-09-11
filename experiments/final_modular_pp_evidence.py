#!/usr/bin/env python3
"""Physical-unit evidence audit for the development-final modular PP routes.

The script never searches PP variants.  It reloads the already frozen prediction
artifacts and compares them with the strongest competitor for which row-aligned
predictions are locally reproducible.  Literature/table-only comparators are kept
separate from this paired analysis.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(LEGACY)]

OUT = ROOT / "results/final_modular_pp_evidence_v1"
SEEDS = tuple(range(42, 47))
RNG_SEED = 20260908


def r2(y: np.ndarray, p: np.ndarray) -> float:
    den = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / den)


def load_npz(path: str):
    return np.load(ROOT / path, allow_pickle=True)


def competitor(name: str, model: str):
    z = load_npz(f"results/extrapolation_competitors_all_v1/{name}_{model}_predictions.npz")
    return np.asarray(z["y"], float), np.asarray(z["groups"]).astype(str), np.asarray(z["prediction"], float)


def nasa_competitor(model: str):
    parts = [load_npz(f"results/extrapolation_competitors_all_v1/nasa_fold{i}_{model}_predictions.npz") for i in range(4)]
    return (
        np.concatenate([np.asarray(z["y"], float) for z in parts]),
        np.concatenate([np.asarray(z["groups"]).astype(str) for z in parts]),
        np.concatenate([np.asarray(z["prediction"], float) for z in parts], axis=1),
    )


def battery_scale_and_controls():
    from pae_boundary_realdata import DATASETS, prepare_dataset
    from pae_shared_battery_nn import BatteryRepresentationScale

    final = load_npz("results/bq_dual_scale_final_replay_v1/predictions.npz")
    controls = load_npz("results/bq_pp_matched_controls_v1/predictions.npz")
    out = {}
    for index, name in enumerate(DATASETS):
        split, audit = prepare_dataset(name)
        scale = BatteryRepresentationScale.fit(split["train"], audit["boundary"]).time_scale
        mask = np.asarray(final["dataset"]) == index
        assert np.array_equal(np.asarray(final["y"])[mask], np.asarray(controls["y"])[mask])
        # Paper-final routing is fixed-bound BQ-PP for Sunwoda/RWTH and
        # support-adaptive dual-scale only for MICH.  Applying the dual-scale
        # replay to all three would audit a non-final executor.
        selected_pp = (
            np.asarray(final["prediction"])[:, mask]
            if name == "mich"
            else np.asarray(controls["bq_pp"])[:, mask]
        )
        out[name] = {
            "y": np.asarray(final["y"])[mask].astype(float) * scale,
            "groups": np.asarray(final["units"])[mask].astype(str),
            "pp": selected_pp.astype(float) * scale,
            "direct_nn": np.asarray(controls["direct_nn"])[:, mask].astype(float) * scale,
        }
    return out


def ncmapss_final():
    from ncmapss_tra_quantile_split import make_tra_hard_split

    preds = []
    reference = None
    for seed in SEEDS:
        z = load_npz(f"results/ncmapss_pp_multiscale_v1/seed{seed}.npz")
        if reference is None:
            reference = z
        preds.append(np.asarray(z["prediction"], float))
    split = make_tra_hard_split(
        ROOT / "data/N-CMAPSS_DS02-006.h5", max_windows_per_unit=1500, random_seed=42
    )
    mask = split.all_windows.hard_extrap_mask(
        split.meta["unit_ids"]["test"], thresholds=split.thresholds
    )
    y = np.asarray(reference["y"], float)[mask]
    groups = np.asarray(reference["units"])[mask].astype(str)
    return y, groups, np.stack(preds)[:, mask]


def build_datasets():
    data = {}

    z = load_npz("results/hust_regime_transport_pp_v1/predictions.npz")
    cy, cg, cp = competitor("hust", "groupdro")
    data["HUST"] = (z["y"], z["groups"], z["transported"], cy, cg, cp, "GroupDRO")

    z = load_npz("results/support_gated_cross_domain_v1/predictions_virkler.npz")
    pp = np.stack([z[f"gated_seed{s}"] for s in SEEDS])
    cy, cg, cp = competitor("virkler", "linear_rff")
    data["Virkler"] = (z["truth"], z["groups"], pp, cy, cg, cp, "linear-tail RBF")

    z = load_npz("results/nasa_causal_multiscale_pp_v1/predictions.npz")
    cy, cg, cp = nasa_competitor("linear_rff")
    data["NASA"] = (z["y"], z["groups"], z["prediction"], cy, cg, cp, "linear-tail RBF")

    batteries = battery_scale_and_controls()
    for name in ("sunwoda", "rwth", "mich"):
        b = batteries[name]
        if name == "sunwoda":
            cy, cg, cp = competitor(name, "linear_rff")
            baseline, label = cp, "linear-tail RBF"
        else:
            cy, cg, baseline = b["y"], b["groups"], b["direct_nn"]
            label = "matched direct NN"
        data[name.upper()] = (b["y"], b["groups"], b["pp"], cy, cg, baseline, label)

    z = load_npz("results/matr_pp_validation_calibration_v1/predictions.npz")
    ft = load_npz("results/matr_ft_validation_calibration_v1/predictions.npz")
    data["MATR2019"] = (z["y"], z["groups"], z["prediction"], ft["y"], ft["groups"], ft["calibrated"], "calibrated FT-Transformer")

    z = load_npz("results/matr_batch2_pp_five_seed_replay/predictions.npz")
    cy, cg, cp = competitor("matr_batch2", "groupdro")
    data["MATR-b2"] = (z["truth"], z["groups"], z["transported"], cy, cg, cp, "GroupDRO (saved predictions)")

    y, groups, pp = ncmapss_final()
    cy, cg, cp = competitor("ncmapss", "monotone")
    data["N-CMAPSS"] = (y, groups, pp, cy, cg, cp, "monotone NN (saved predictions)")
    return data


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    observed = abs(float(np.mean(values)))
    if len(values) <= 16:
        null = [abs(float(np.mean(values * np.asarray(s)))) for s in product((-1.0, 1.0), repeat=len(values))]
        return float(np.mean(np.asarray(null) >= observed - 1e-15))
    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    return float((1 + np.count_nonzero(np.abs(np.mean(signs * values, axis=1)) >= observed)) / 100_001)


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * len(p) / (rank + 1))
        q[idx] = running
    return q.tolist()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 30_000):
    values = np.asarray(values, float)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    datasets = build_datasets()
    rows = []
    results = []
    for name, (y, groups, pp, cy, cg, baseline, label) in datasets.items():
        y = np.asarray(y, float)
        groups = np.asarray(groups).astype(str)
        pp = np.asarray(pp, float)
        cy = np.asarray(cy, float)
        cg = np.asarray(cg).astype(str)
        baseline = np.asarray(baseline, float)
        assert y.shape == cy.shape and np.allclose(y, cy), f"target mismatch: {name}"
        assert np.array_equal(groups, cg), f"group/order mismatch: {name}"
        assert pp.shape == baseline.shape == (5, len(y)), f"seed shape mismatch: {name}"
        pp_ensemble, base_ensemble = pp.mean(axis=0), baseline.mean(axis=0)
        unit_effects = []
        unit_log_ratios = []
        for unit in np.unique(groups):
            mask = groups == unit
            pp_rmse = math.sqrt(float(np.mean((y[mask] - pp_ensemble[mask]) ** 2)))
            base_rmse = math.sqrt(float(np.mean((y[mask] - base_ensemble[mask]) ** 2)))
            effect = 1.0 - pp_rmse / max(base_rmse, 1e-12)
            log_ratio = math.log(max(base_rmse, 1e-12) / max(pp_rmse, 1e-12))
            unit_effects.append(effect)
            unit_log_ratios.append(log_ratio)
            rows.append({"dataset": name, "unit": unit, "n": int(mask.sum()), "pp_rmse": pp_rmse, "baseline_rmse": base_rmse, "relative_rmse_gain": effect, "log_rmse_ratio": log_ratio})
        unit_effects = np.asarray(unit_effects)
        unit_log_ratios = np.asarray(unit_log_ratios)
        pp_seed_r2 = [r2(y, p) for p in pp]
        base_seed_r2 = [r2(y, p) for p in baseline]
        results.append({
            "dataset": name,
            "baseline": label,
            "n_rows": len(y),
            "n_units": len(unit_effects),
            "pp_ensemble_r2": r2(y, pp_ensemble),
            "baseline_ensemble_r2": r2(y, base_ensemble),
            "r2_gap": r2(y, pp_ensemble) - r2(y, base_ensemble),
            "pp_seed_r2_mean": float(np.mean(pp_seed_r2)),
            "pp_seed_r2_sd": float(np.std(pp_seed_r2, ddof=1)),
            "baseline_seed_r2_mean": float(np.mean(base_seed_r2)),
            "baseline_seed_r2_sd": float(np.std(base_seed_r2, ddof=1)),
            "mean_unit_relative_rmse_gain": float(np.mean(unit_effects)),
            "unit_bootstrap_ci95": bootstrap_ci(unit_effects, rng),
            "mean_unit_log_rmse_ratio": float(np.mean(unit_log_ratios)),
            "unit_log_ratio_bootstrap_ci95": bootstrap_ci(unit_log_ratios, rng),
            "units_won": int(np.count_nonzero(unit_effects > 0)),
            "sign_flip_p_two_sided": exact_sign_flip(unit_log_ratios),
            "unit_effects": unit_effects.tolist(),
            "unit_log_ratios": unit_log_ratios.tolist(),
        })
    q = bh_adjust([x["sign_flip_p_two_sided"] for x in results])
    for item, value in zip(results, q):
        item["bh_q"] = value

    effects = [np.asarray(x["unit_effects"], float) for x in results]
    log_effects = [np.asarray(x["unit_log_ratios"], float) for x in results]
    hierarchical = []
    hierarchical_log = []
    for _ in range(30_000):
        selected = rng.integers(0, len(effects), len(effects))
        means = [float(np.mean(rng.choice(effects[i], size=len(effects[i]), replace=True))) for i in selected]
        hierarchical.append(float(np.mean(means)))
        log_means = [float(np.mean(rng.choice(log_effects[i], size=len(log_effects[i]), replace=True))) for i in selected]
        hierarchical_log.append(float(np.mean(log_means)))
    dataset_means = np.asarray([np.mean(x) for x in effects])
    aggregate = {
        "datasets": len(results),
        "pp_ensemble_wins": int(sum(x["r2_gap"] > 0 for x in results)),
        "dataset_sign_test_two_sided": float(2 / (2 ** len(results))),
        "equal_dataset_mean_unit_relative_rmse_gain": float(np.mean(dataset_means)),
        "hierarchical_bootstrap_ci95": [float(x) for x in np.quantile(hierarchical, [0.025, 0.975])],
        "equal_dataset_mean_unit_log_rmse_ratio": float(np.mean([np.mean(x) for x in log_effects])),
        "hierarchical_log_ratio_ci95": [float(x) for x in np.quantile(hierarchical_log, [0.025, 0.975])],
        "geometric_mean_rmse_reduction": float(1.0 - math.exp(-np.mean([np.mean(x) for x in log_effects]))),
        "total_physical_units": int(sum(x["n_units"] for x in results)),
        "note": "Paired inference uses strongest locally stored row-aligned competitor predictions, not table-only comparators.",
    }
    payload = {
        "status": "retrospective development-final modular PP evidence; no new PP variant selected",
        "protocol": {
            "primary_metric": "pooled R2",
            "paired_effect": "1 - unit_RMSE(PP) / unit_RMSE(baseline)",
            "seeds": list(SEEDS),
            "bootstrap_repetitions": 30_000,
            "multiple_testing": "Benjamini-Hochberg across dataset-level unit sign-flip tests",
        },
        "aggregate": aggregate,
        "datasets": results,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUT / "unit_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2))
    for item in results:
        print(f"{item['dataset']:10s} PP={item['pp_ensemble_r2']:.3f} base={item['baseline_ensemble_r2']:.3f} gap={item['r2_gap']:+.3f} units={item['units_won']}/{item['n_units']} q={item['bh_q']:.4g}")


if __name__ == "__main__":
    main()
