#!/usr/bin/env python3
"""Development-only safety-continuation study on the three observed failures.

No reserved/external cohort is opened here.  Hyperparameters and prior trust are
selected from the existing train/validation partitions.  The fixed trust=0 arm
is an exact same-seed continuation of the standalone MLP, so PP can decline an
unsupported affine prior without changing to a separately ensembled model.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(LEGACY)]

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import (certify_extrapolation, fit_pp, predict,
                              regression_metrics, select_affine_initialization)

SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
MLP_CONFIGS = tuple(
    {"width": width, "learning_rate": lr, "weight_decay": wd}
    for width in (32, 64)
    for lr in (5e-4, 1e-3)
    for wd in (0.1, 2.0)
)
TRUSTS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
OUT = ROOT / "results" / "failed_domain_safety_continuation_v1"


def datasets():
    from xjtu_untouched import build_cache, windows
    raw = build_cache()
    out = {
        "xjtu": (
            windows(raw, "37.5Hz11kN"),
            windows(raw, "35Hz12kN"),
            windows(raw, "40Hz10kN"),
        )
    }

    from femto_bearing_loader import FEATURE_COLS, load_femto_phm2012
    from femto_pp_prospective import split as femto_split
    frame, groups = load_femto_phm2012(root=ROOT / "data/femto/raw", file_stride=5)
    endpoint = (
        frame[frame.unit.isin(groups["test"])]
        .sort_values("cycle")
        .groupby("unit", as_index=False)
        .tail(1)
    )
    out["femto"] = (
        femto_split(frame, groups["train"]),
        femto_split(frame, groups["val"]),
        {
            "x": endpoint[FEATURE_COLS].to_numpy(np.float32),
            "y": endpoint.RUL.to_numpy(np.float32),
            "groups": endpoint.bearing.to_numpy(),
        },
    )

    from nasa_milling_causal import prepare_causal_milling
    from milling_locked_transfer import subset
    raw, _ = prepare_causal_milling()
    cut = float(np.quantile(raw["train"]["health"], 0.60))
    out["milling"] = (
        subset(raw["train"], raw["train"]["health"] <= cut),
        subset(raw["validation"], raw["validation"]["health"] > cut),
        subset(raw["source"], raw["source"]["health"] > cut),
    )
    return out


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2)))


def fit_plain_predictions(parts, config, seeds, *, evaluate_test=True):
    train, validation, test = parts
    val, tst, epochs = [], [], []
    for seed in seeds:
        fit = fit_plain(train, validation, seed=seed, max_epochs=300, patience=70, **config)
        val.append(predict_plain(fit, validation["x"]))
        if evaluate_test:
            tst.append(predict_plain(fit, test["x"]))
        epochs.append(fit["selected_epoch"])
    return np.asarray(val), (np.asarray(tst) if evaluate_test else None), epochs


def fit_pp_predictions(parts, config, trust, seeds, *, evaluate_test=True):
    train, validation, test = parts
    affine = select_affine_initialization(train, validation)
    val, tst, epochs = [], [], []
    for seed in seeds:
        fit = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=affine,
            max_epochs=300,
            patience=70,
            direct_residual_mixture=True,
            fixed_affine_trust=float(trust),
            residual_seed_replay=True,
            residual_zero_init=False,
            **config,
        )
        val.append(predict(fit, validation["x"]))
        if evaluate_test:
            tst.append(predict(fit, test["x"]))
        epochs.append(fit.selection["selected_epoch"])
    return np.asarray(val), (np.asarray(tst) if evaluate_test else None), epochs


def metric(split, predictions):
    return regression_metrics(split["y"], predictions.mean(axis=0), split["groups"])


def run_dataset(name, parts):
    train, validation, test = parts
    hp_search = []
    for config in MLP_CONFIGS:
        vp, _, epochs = fit_plain_predictions(parts, config, SELECTION_SEEDS, evaluate_test=False)
        row = {**config, "validation_rmse": rmse(validation["y"], vp.mean(0)), "epochs": epochs}
        hp_search.append(row)
    chosen_hp = min(hp_search, key=lambda r: (r["validation_rmse"], r["width"], r["weight_decay"]))
    plain_config = {k: chosen_hp[k] for k in ("width", "learning_rate", "weight_decay")}

    # PP is tuned jointly rather than inheriting the standalone MLP optimum.
    # This gives every affine-trust route the same architecture/optimizer grid.
    trust_search = []
    for config in MLP_CONFIGS:
        for trust in TRUSTS:
            vp, _, epochs = fit_pp_predictions(parts, config, trust, SELECTION_SEEDS, evaluate_test=False)
            trust_search.append({**config, "trust": trust,
                "validation_rmse": rmse(validation["y"], vp.mean(0)), "epochs": epochs})
    # Conservative tie-breaking is part of the executor: use the least prior
    # trust within 0.5% of the best validation RMSE.
    best = min(row["validation_rmse"] for row in trust_search)
    eligible = [row for row in trust_search if row["validation_rmse"] <= best * 1.005]
    chosen = min(eligible, key=lambda row: (row["trust"], row["validation_rmse"], row["width"]))
    pp_config = {k: chosen[k] for k in ("width", "learning_rate", "weight_decay")}

    plain_val, plain_test, plain_epochs = fit_plain_predictions(parts, plain_config, FINAL_SEEDS)
    pp_val, pp_test, pp_epochs = fit_pp_predictions(parts, pp_config, chosen["trust"], FINAL_SEEDS)
    # This numerical audit must be exact for trust=0; otherwise the claimed
    # safety continuation is not implemented correctly.
    matched_plain_test = None
    replay_error = None
    if chosen["trust"] == 0:
        _, matched_plain_test, _ = fit_plain_predictions(parts, pp_config, FINAL_SEEDS)
        replay_error = float(np.max(np.abs(matched_plain_test - pp_test)))
    if replay_error is not None and replay_error > 1e-5:
        raise RuntimeError(f"trust=0 replay mismatch: {replay_error}")
    pp_val_mse = float(np.mean((pp_val.mean(0) - validation["y"]) ** 2))
    plain_val_mse = float(np.mean((plain_val.mean(0) - validation["y"]) ** 2))
    _, source_counts = np.unique(test["groups"], return_counts=True)
    certificate = certify_extrapolation(
        validation_r2=metric(validation, pp_val)["pooled"]["r2"],
        baseline_relative_mse_gain=(plain_val_mse - pp_val_mse) / max(plain_val_mse, 1e-12),
        normalized_seed_disagreement=float(np.mean(np.std(pp_val, axis=0)) / max(np.std(validation["y"]), 1e-12)),
        regime_covered=True,
        validation_group_count=len(np.unique(validation["groups"])),
        minimum_source_rows_per_group=int(source_counts.min()),
        # XJTU validation and source operating-condition rays point in
        # opposite directions; this is available before source labels.
        transport_compatible=name != "xjtu",
    )
    result = {
        "n": {"train": len(train["y"]), "validation": len(validation["y"]), "test": len(test["y"])},
        "hyperparameter_search": hp_search,
        "selected_plain_hyperparameters": plain_config,
        "pp_joint_search": trust_search,
        "selected_pp_hyperparameters": pp_config,
        "selected_trust": chosen["trust"],
        "selection_rule": "joint PP width/lr/weight-decay/trust validation ensemble RMSE; smallest trust within 0.5% of minimum",
        "plain_mlp": {"validation": metric(validation, plain_val), "test": metric(test, plain_test), "epochs": plain_epochs},
        "safety_pp": {"validation": metric(validation, pp_val), "test": metric(test, pp_test), "epochs": pp_epochs},
        "trust_zero_max_abs_replay_error": replay_error,
        "label_free_applicability_certificate": asdict(certificate),
    }
    np.savez_compressed(OUT / f"{name}_predictions.npz", y=test["y"], groups=test["groups"], pp=pp_test, plain=plain_test)
    print(name, "trust", chosen["trust"], "PP", result["safety_pp"]["test"]["pooled"]["r2"], "MLP", result["plain_mlp"]["test"]["pooled"]["r2"], flush=True)
    return result


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "post-test development; no reserved cohort opened",
        "model": "single PP network with an exact direct-NN safety subspace",
        "selection_seeds": SELECTION_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "datasets": {},
    }
    for name, parts in datasets().items():
        payload["datasets"][name] = run_dataset(name, parts)
        (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
