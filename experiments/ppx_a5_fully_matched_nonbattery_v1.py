#!/usr/bin/env python3
"""A5: prior-OFF vs prior-ON with identical nonlinear capacity.

The only forward difference is the frozen affine prior contribution.  Splits,
features, nonlinear architecture, initialization replay, optimizer, training
budget, checkpoint rule, and seeds are shared within each setting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ppx_a5_fully_matched_nonbattery_v1"
REPORT = ROOT / "PPX_A5_FULLY_MATCHED_NONBATTERY_V1_KO.md"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]

from group_robust_pp import batch2  # noqa: E402
from ncmapss_pp_multiscale import as_rows, make_features  # noqa: E402
from pp_extrapolation import (  # noqa: E402
    fit_latent_regime_pp,
    fit_pp,
    predict,
    predict_latent_regime,
    select_affine_initialization,
)
from ppx_final_ablation_statistics import compare  # noqa: E402
from run_affine_tail_external_three import prepare_hust  # noqa: E402

SEEDS = tuple(range(42, 47))


def fit_pair(train, validation, test_x, *, seed: int, config: dict):
    affine = select_affine_initialization(train, validation)
    common = dict(
        seed=seed,
        affine_selection=affine,
        residual_zero_init=False,
        residual_seed_replay=True,
        **config,
    )
    direct = fit_pp(
        train,
        validation,
        direct_residual_mixture=True,
        fixed_affine_trust=0.0,
        **common,
    )
    prior = fit_pp(train, validation, **common)
    return {
        "direct_val": predict(direct, validation["x"]),
        "prior_val": predict(prior, validation["x"]),
        "direct_test": predict(direct, test_x),
        "prior_test": predict(prior, test_x),
        "direct_epoch": direct.selection["selected_epoch"],
        "prior_epoch": prior.selection["selected_epoch"],
    }


def run_standard(name: str, train: dict, validation: dict, test: dict, config: dict, seeds) -> dict:
    runs = []
    for seed in seeds:
        row = fit_pair(train, validation, test["x"], seed=seed, config=config)
        runs.append(row)
        print(name, seed, row["direct_epoch"], row["prior_epoch"], flush=True)
    return {
        "name": name,
        "val_y": validation["y"],
        "val_groups": validation["groups"],
        "test_y": test["y"],
        "test_groups": test["groups"],
        "direct_val": np.asarray([row["direct_val"] for row in runs]),
        "prior_val": np.asarray([row["prior_val"] for row in runs]),
        "direct_test": np.asarray([row["direct_test"] for row in runs]),
        "prior_test": np.asarray([row["prior_test"] for row in runs]),
        "epochs": [{"seed": seed, "direct": row["direct_epoch"], "prior": row["prior_epoch"]} for seed, row in zip(seeds, runs)],
    }


def run_ncmapss(seeds) -> dict:
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_tra_quantile_split import make_tra_hard_split

    split = make_tra_hard_split(ROOT / "data/N-CMAPSS_DS02-006.h5", max_windows_per_unit=1500, random_seed=42)
    names = list(FEATURE_COLS)
    train = as_rows(split.train, names, "multiscale")
    validation = as_rows(split.val, names, "multiscale")
    affine = select_affine_initialization(train, validation)
    all_x = make_features(split.all_windows, names, "multiscale")
    val_mask = split.all_windows.hard_extrap_mask(split.meta["unit_ids"]["val"], thresholds=split.thresholds)
    test_mask = split.all_windows.hard_extrap_mask(split.meta["unit_ids"]["test"], thresholds=split.thresholds)
    y = np.asarray(split.all_windows.y, float)
    groups = np.asarray(split.all_windows.units)
    store = {key: [] for key in ("direct_val", "prior_val", "direct_test", "prior_test")}
    epochs = []
    for seed in seeds:
        common = dict(
            seed=seed,
            affine_selection=affine,
            max_epochs=300,
            patience=70,
            separation_weight=0.01,
            gate_weight=0.0,
        )
        direct = fit_latent_regime_pp(train, validation, affine_scale=0.0, **common)
        prior = fit_latent_regime_pp(train, validation, affine_scale=1.0, **common)
        store["direct_val"].append(predict_latent_regime(direct, all_x[val_mask]))
        store["prior_val"].append(predict_latent_regime(prior, all_x[val_mask]))
        store["direct_test"].append(predict_latent_regime(direct, all_x[test_mask]))
        store["prior_test"].append(predict_latent_regime(prior, all_x[test_mask]))
        epochs.append({"seed": seed, "direct": direct.selection["selected_epoch"], "prior": prior.selection["selected_epoch"]})
        print("N-CMAPSS", seed, epochs[-1]["direct"], epochs[-1]["prior"], flush=True)
    return {
        "name": "N-CMAPSS",
        "val_y": y[val_mask],
        "val_groups": groups[val_mask],
        "test_y": y[test_mask],
        "test_groups": groups[test_mask],
        **{key: np.asarray(value) for key, value in store.items()},
        "epochs": epochs,
    }


def summarize(setting: dict) -> dict:
    y, groups = setting["test_y"], setting["test_groups"]
    direct, prior = setting["direct_test"], setting["prior_test"]
    stats = compare(setting["name"], "A5 matched prior conditioning", y, groups, direct, prior, status="fully_matched")
    return {
        "setting": setting["name"],
        "direct_r2": float(r2_score(y, direct.mean(0))),
        "prior_residual_r2": float(r2_score(y, prior.mean(0))),
        "delta_r2": float(r2_score(y, prior.mean(0)) - r2_score(y, direct.mean(0))),
        "mean_unit_rmse_reduction": stats["mean_unit_rmse_reduction"],
        "unit_bootstrap_ci95": stats["unit_bootstrap_ci95"],
        "unit_signflip_p_two_sided": stats["unit_signflip_p_two_sided"],
        "epochs": setting["epochs"],
    }


def write_report(payload: dict) -> None:
    lines = [
        "# PP-X A5 — Fully Matched Non-Battery Prior Conditioning", "",
        "Prior-OFF와 Prior-ON은 동일 split, feature, seed, nonlinear architecture, initialization replay, optimizer, budget, checkpoint rule을 사용했다. 차이는 frozen affine prior contribution뿐이다.", "",
        "| Setting | No-prior NN R² | Prior+same residual R² | ΔR² | Unit RMSE reduction 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["settings"]:
        lo, hi = row["unit_bootstrap_ci95"]
        lines.append(f"| {row['setting']} | {row['direct_r2']:.3f} | {row['prior_residual_r2']:.3f} | {row['delta_r2']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", f"Positive/negative settings: **{payload['positive_settings']}/{payload['negative_settings']}**", "",
              "This is a retrospective matched ablation, not prospective confirmation.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--datasets", nargs="*", choices=("hust", "matr", "ncmapss"), default=("hust", "matr", "ncmapss"))
    args = parser.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    settings = []
    if "hust" in args.datasets:
        tr, va, te = prepare_hust()[:3]
        settings.append(run_standard("HUST", tr, va, te, {"max_epochs": 300, "patience": 70, "width": 64}, seeds))
    if "matr" in args.datasets:
        tr, va, te = batch2()
        settings.append(run_standard("MATR-b2", tr, va, te, {"max_epochs": 400, "patience": 80, "width": 32, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.05}, seeds))
    if "ncmapss" in args.datasets:
        settings.append(run_ncmapss(seeds))
    for setting in settings:
        np.savez_compressed(OUT / f"{setting['name'].lower().replace('-', '_')}.npz", **{k: v for k, v in setting.items() if k not in {"name", "epochs"}})
    rows = [summarize(setting) for setting in settings]
    payload = {
        "experiment": "ppx_a5_fully_matched_nonbattery_v1",
        "smoke": args.smoke,
        "seeds": list(seeds),
        "matching": "same split/features/nonlinear architecture/init replay/optimizer/budget/checkpoint; affine prior scale 0 vs 1",
        "settings": rows,
        "positive_settings": sum(row["mean_unit_rmse_reduction"] > 0 for row in rows),
        "negative_settings": sum(row["mean_unit_rmse_reduction"] < 0 for row in rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    name = "smoke_results.json" if args.smoke else "results.json"
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        write_report(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
