#!/usr/bin/env python3
"""Locked PP-model-level prospective replay on C-MAPSS FD001 and FD003."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(LEGACY)]

from cmapss_fd002_loader import FEATURE_COLS, load_fd_test, load_fd_train, load_rul_truth
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, regression_metrics, select_affine_initialization
from pp_extrapolation.model import equal_group_weights, fit_feature_scale, transform_features

SEEDS = (42, 43, 44, 45, 46)
OUT = ROOT / "results" / "cmapss_fd001_fd003_pp_cohort_v1"


def causal_features(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices, units, cycles = [], [], []
    for unit, part in frame.groupby("unit", sort=True):
        part = part.sort_values("cycle")
        raw = part[FEATURE_COLS].to_numpy(np.float64)
        cycle = part["cycle"].to_numpy(np.float64)
        n = len(part); mean = np.empty_like(raw); std = np.empty_like(raw); slope = np.empty_like(raw)
        for i in range(n):
            lo = max(0, i - 29); window = raw[lo:i + 1]
            mean[i] = window.mean(0); std[i] = window.std(0)
            dt = max(cycle[i] - cycle[lo], 1.0)
            slope[i] = (raw[i] - raw[lo]) / dt
        matrices.append(np.column_stack([raw, mean, std, slope, cycle, np.log1p(cycle)]))
        units.extend([f"{int(unit)}"] * n); cycles.append(cycle)
    return np.concatenate(matrices), np.asarray(units), np.concatenate(cycles)


def prepare(fd: str):
    train_frame = load_fd_train(fd).copy()
    # Replace the loader's standard capped target with physical uncapped RUL.
    train_frame["RUL"] = train_frame.groupby("unit")["cycle"].transform("max") - train_frame["cycle"]
    x, groups, _ = causal_features(train_frame)
    y = train_frame["RUL"].to_numpy(np.float64)
    ids = np.asarray(sorted(train_frame["unit"].unique()))
    cut = int(np.floor(0.8 * len(ids))); train_ids, validation_ids = ids[:cut], ids[cut:]
    tr_mask = np.isin(groups.astype(int), train_ids); va_mask = ~tr_mask
    keep = x[tr_mask].std(0) > 1e-10
    train = {"x": x[tr_mask][:, keep], "y": y[tr_mask], "groups": groups[tr_mask]}
    validation = {"x": x[va_mask][:, keep], "y": y[va_mask], "groups": groups[va_mask]}
    full = {"x": x[:, keep], "y": y, "groups": groups}

    test_frame = load_fd_test(fd)
    tx, tgroups, _ = causal_features(test_frame)
    endpoint = np.r_[np.flatnonzero(tgroups[1:] != tgroups[:-1]), len(tgroups) - 1]
    test_x = tx[endpoint][:, keep]
    test_groups = np.asarray([f"{fd}-test-{u}" for u in tgroups[endpoint]])
    return train, validation, full, test_x, test_groups, keep, train_ids, validation_ids


def affine_prediction(fit, x):
    value = torch.as_tensor(transform_features(x, fit.center, fit.scale), dtype=torch.float32)
    fit.model.eval()
    with torch.no_grad(): estimate = fit.model.affine(value).squeeze(1).numpy() * fit.target_scale
    return np.clip(estimate, 0.0, fit.target_scale)


def run_subset(fd: str) -> dict:
    train, validation, full, test_x, test_groups, keep, train_ids, validation_ids = prepare(fd)
    selection = select_affine_initialization(train, validation)
    pp_predictions, affine_predictions, mlp_predictions, run_meta = [], [], [], []
    for seed in SEEDS:
        selected_pp = fit_pp(train, validation, seed=seed, affine_selection=selection,
                             max_epochs=300, patience=50)
        pp_epoch = max(int(selected_pp.selection["selected_epoch"]), 1)
        full_affine = select_affine_initialization(full, full, alphas=(selection["selected_alpha"],))
        final_pp = fit_pp(full, full, seed=seed, affine_selection=full_affine,
                          max_epochs=pp_epoch, patience=pp_epoch + 1)
        selected_mlp = fit_plain(train, validation, seed=seed, max_epochs=300, patience=50)
        mlp_epoch = max(int(selected_mlp["selected_epoch"]), 1)
        final_mlp = fit_plain(full, full, seed=seed, max_epochs=mlp_epoch,
                              patience=mlp_epoch + 1, restore_best=False)
        pp_predictions.append(predict(final_pp, test_x))
        affine_predictions.append(affine_prediction(final_pp, test_x))
        mlp_predictions.append(predict_plain(final_mlp, test_x))
        run_meta.append({"seed": seed, "pp_epoch": pp_epoch, "mlp_epoch": mlp_epoch})

    # Protocol requirement: official test labels are opened only after every
    # architecture choice and prediction above has been frozen in memory.
    truth = np.asarray(load_rul_truth(fd), dtype=np.float64).reshape(-1)
    if len(truth) != len(test_x):
        raise RuntimeError(f"{fd}: {len(truth)} truths for {len(test_x)} endpoints")
    pp = np.asarray(pp_predictions); affine = np.asarray(affine_predictions); mlp = np.asarray(mlp_predictions)
    for meta, p, a, m in zip(run_meta, pp, affine, mlp):
        meta["pp"] = regression_metrics(truth, p, test_groups)
        meta["affine"] = regression_metrics(truth, a, test_groups)
        meta["mlp"] = regression_metrics(truth, m, test_groups)
    result = {
        "fd": fd, "train_units": train_ids.tolist(), "validation_units": validation_ids.tolist(),
        "n_features": int(keep.sum()), "n_test_engines": len(truth),
        "selected_alpha": float(selection["selected_alpha"]), "runs": run_meta,
        "ensemble": {
            "pp": regression_metrics(truth, pp.mean(0), test_groups),
            "affine": regression_metrics(truth, affine.mean(0), test_groups),
            "plain_mlp": regression_metrics(truth, mlp.mean(0), test_groups),
        },
    }
    result["success"] = bool(result["ensemble"]["pp"]["pooled"]["r2"] > 0 and
                             result["ensemble"]["pp"]["pooled"]["rmse"] <
                             result["ensemble"]["plain_mlp"]["pooled"]["rmse"])
    np.savez_compressed(OUT / f"{fd.lower()}_predictions.npz", y=truth, groups=test_groups,
                        pp=pp, affine=affine, mlp=mlp)
    print(fd, result["ensemble"], result["success"], flush=True)
    return result


def main() -> None:
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    subsets = {fd: run_subset(fd) for fd in ("FD001", "FD003")}
    payload = {
        "status": "PP-model-level prospective; dataset previously used by PAE",
        "protocol_commit": "0b9290e", "subsets": subsets,
        "combined_success": all(value["success"] for value in subsets.values()),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v["ensemble"] for k, v in subsets.items()}, indent=2))


if __name__ == "__main__":
    main()
