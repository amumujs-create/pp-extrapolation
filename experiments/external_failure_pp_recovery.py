#!/usr/bin/env python3
"""Development-only PP recovery study for failed NASA/UCF and CALCE cohorts.

HNEI is deliberately absent: its locked successful result is never loaded or rerun.
Model families are selected by validation ensemble RMSE, then reported once on the
already-observed test partitions.  Consequently these are development results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from calce_external_eval import load_series as load_calce, rows as battery_rows
from nasa_alt_external_eval import extract as load_nasa, feature_rows as nasa_rows
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import (fit_pp, regression_metrics,
                              select_affine_initialization, support_distance)
from pp_extrapolation.residual_calibration import combine_residual_gain
from pp_extrapolation.support_gate import predict_components

OUT = ROOT / "results" / "external_failure_pp_recovery_v1"
SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# Every arm retains PP's affine + neural-residual decomposition.  A zero anchor
# allows the affine path to move, while positive anchors softly preserve the
# train-only ridge initialization.
CONFIGS = (
    {"name": "frozen_default", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0},
    {"name": "relaxed_a0", "affine_anchor_weight": 0.0, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0},
    {"name": "relaxed_a001", "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0},
    {"name": "relaxed_a01", "affine_anchor_weight": 0.1, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0},
    {"name": "relaxed_lowwd", "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 5e-4, "weight_decay": 0.1, "residual_decay": 0.0},
    {"name": "relaxed_w64", "affine_anchor_weight": 0.01, "width": 64, "learning_rate": 5e-4, "weight_decay": 0.5, "residual_decay": 0.0},
    {"name": "relaxed_fast", "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 1e-3, "weight_decay": 0.5, "residual_decay": 0.0},
    {"name": "frozen_decay025", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.25},
    {"name": "relaxed_decay025", "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 5e-4, "weight_decay": 0.5, "residual_decay": 0.25},
    {"name": "trust_gate", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False},
    {"name": "trust_gate_relaxed", "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 5e-4, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False},
    {"name": "trust_gate_w64", "affine_anchor_weight": 0.01, "width": 64, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False},
    {"name": "trust_gate_10pct", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.1},
    {"name": "trust_gate_01pct", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.01},
    {"name": "trust_gate_10pct_lowwd", "affine_anchor_weight": None, "width": 64, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.1},
    {"name": "moe_trust10", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.1, "direct_residual_mixture": True},
    {"name": "moe_trust25", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.25, "direct_residual_mixture": True},
    {"name": "moe_trust50", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.5, "direct_residual_mixture": True},
    {"name": "moe_trust75", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.75, "direct_residual_mixture": True},
    {"name": "moe_w64_lowwd", "affine_anchor_weight": None, "width": 64, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.25, "direct_residual_mixture": True},
    {"name": "clock_frozen", "clock_residual": True, "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0},
    {"name": "clock_relaxed", "clock_residual": True, "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 1e-3, "weight_decay": 0.5, "residual_decay": 0.0},
    {"name": "clock_gate", "clock_residual": True, "affine_anchor_weight": None, "width": 64, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.25, "direct_residual_mixture": True},
    {"name": "history_frozen", "history_basis": True, "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0},
    {"name": "history_relaxed", "history_basis": True, "affine_anchor_weight": 0.01, "width": 32, "learning_rate": 1e-3, "weight_decay": 0.5, "residual_decay": 0.0},
    {"name": "history_gate", "history_basis": True, "affine_anchor_weight": None, "width": 64, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.25},
    {"name": "history_moe", "history_basis": True, "affine_anchor_weight": None, "width": 64, "learning_rate": 1e-3, "weight_decay": 0.1, "residual_decay": 0.0, "learned_affine_gate": True, "residual_zero_init": False, "affine_gate_initial_trust": 0.25, "direct_residual_mixture": True},
    {"name": "safe_nn", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.0},
    {"name": "safe_pp02", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.02},
    {"name": "safe_pp05", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.05},
    {"name": "safe_pp10", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.1},
    {"name": "safe_pp20", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.2},
    {"name": "safe_pp40", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.4},
    {"name": "safe_exact_nn", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.0, "residual_seed_replay": True, "max_epochs": 300, "patience": 70},
    {"name": "safe_exact_pp02", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.02, "residual_seed_replay": True, "max_epochs": 300, "patience": 70},
    {"name": "safe_exact_pp05", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.05, "residual_seed_replay": True, "max_epochs": 300, "patience": 70},
    {"name": "safe_exact_pp10", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.1, "residual_seed_replay": True, "max_epochs": 300, "patience": 70},
    {"name": "safe_exact_pp20", "affine_anchor_weight": None, "width": 32, "learning_rate": 5e-4, "weight_decay": 2.0, "residual_decay": 0.0, "residual_zero_init": False, "direct_residual_mixture": True, "fixed_affine_trust": 0.2, "residual_seed_replay": True, "max_epochs": 300, "patience": 70},
)


def prepare(name: str):
    if name == "nasa_alt":
        series = load_nasa(); eligible = sorted(k for k, v in series.items() if len(v) >= 10)
        a = int(.6 * len(eligible)); b = a + int(.2 * len(eligible)); ids = {"train": eligible[:a], "validation": eligible[a:b], "test": eligible[b:]}
        scale = max(series[u][-1]["time"] - series[u][0]["time"] for u in ids["train"])
        return ids, nasa_rows(series, ids["train"], "train", scale), nasa_rows(series, ids["validation"], "eval", scale), nasa_rows(series, ids["test"], "eval", scale), scale / 86400.0
    series = load_calce(); eligible = sorted(k for k, v in series.items() if len(v["cycle"]) >= 20)
    a = int(.6 * len(eligible)); b = a + int(.2 * len(eligible)); ids = {"train": eligible[:a], "validation": eligible[a:b], "test": eligible[b:]}
    scale = max(series[u]["cycle"][-1] - series[u]["cycle"][0] for u in ids["train"])
    return ids, battery_rows(series, ids["train"], "train", scale), battery_rows(series, ids["validation"], "eval", scale), battery_rows(series, ids["test"], "eval", scale), scale


def fit_config(config, train, validation, test, seeds, elapsed_scale):
    clock = bool(config.get("clock_residual", False))
    history = bool(config.get("history_basis", False))
    def history_rows(rows):
        if not history: return rows
        x=np.asarray(rows["x"],dtype=np.float32);recent=x[:,2:5];elapsed=np.maximum(x[:,1],1e-4);global_rate=x[:,7]/elapsed
        extra=np.column_stack([np.median(recent,axis=1),np.std(recent,axis=1),np.min(recent,axis=1),np.max(recent,axis=1),global_rate,
                               np.clip(np.median(recent,axis=1)/(global_rate+np.sign(global_rate)*1e-4+1e-4),-20,20)])
        return {**rows,"x":np.column_stack([x,extra]).astype(np.float32)}
    train,validation,test=map(history_rows,(train,validation,test))
    def clock_rows(rows):
        return {**rows, "y": np.asarray(rows["y"]) + np.asarray(rows["x"])[:, 1] * elapsed_scale} if clock else rows
    fit_train, fit_validation = clock_rows(train), clock_rows(validation)
    affine = select_affine_initialization(fit_train, fit_validation)
    dv, _ = support_distance(train["x"][:, [0]], validation["x"][:, [0]])
    dt, _ = support_distance(train["x"][:, [0]], test["x"][:, [0]])
    val_a, val_r, test_a, test_r, fits = [], [], [], [], []
    for seed in seeds:
        kw = {k: v for k, v in config.items() if k not in ("name", "clock_residual", "history_basis", "max_epochs", "patience")}
        fit = fit_pp(fit_train, fit_validation, seed=seed, affine_selection=affine,
                     max_epochs=int(config.get("max_epochs",450)), patience=int(config.get("patience",90)), **kw)
        av, rv = predict_components(fit, validation["x"]); at, rt = predict_components(fit, test["x"])
        if config["residual_decay"]:
            rv *= np.exp(-config["residual_decay"] * dv); rt *= np.exp(-config["residual_decay"] * dt)
        val_a.append(av); val_r.append(rv); test_a.append(at); test_r.append(rt); fits.append(fit)
    val_a, val_r, test_a, test_r = map(np.asarray, (val_a, val_r, test_a, test_r))
    # Gain is an internal residual-confidence parameter; it does not combine PP
    # with a separately trained estimator.
    candidates = []
    gain_grid = (1.0,) if config["name"].startswith("safe_exact") else GAINS
    cap=max(f.target_scale for f in fits)
    def ensemble(parts_a,parts_r,gain,rows):
        predictions=[]
        for a,r in zip(parts_a,parts_r):
            p=combine_residual_gain(a,r,gain=gain,output_cap=cap)
            if clock: p=np.clip(p-np.asarray(rows["x"])[:,1]*elapsed_scale,0,None)
            predictions.append(p)
        return np.mean(predictions,axis=0)
    for gain in gain_grid:
        pred=ensemble(val_a,val_r,gain,validation)
        candidates.append({"gain": gain, "validation_rmse": float(np.sqrt(np.mean((pred - validation["y"]) ** 2)))})
    chosen = min(candidates, key=lambda x: (x["validation_rmse"], abs(x["gain"] - 1)))
    val_pred=ensemble(val_a,val_r,chosen["gain"],validation)
    test_pred=ensemble(test_a,test_r,chosen["gain"],test)
    seeded_test=[]
    for a,r in zip(test_a,test_r):
        p=combine_residual_gain(a,r,gain=chosen["gain"],output_cap=cap)
        if clock: p=np.clip(p-np.asarray(test["x"])[:,1]*elapsed_scale,0,None)
        seeded_test.append(regression_metrics(test["y"],p,test["groups"]))
    return {"config": config, "gain_candidates": candidates, "selected_gain": chosen["gain"],
            "validation": regression_metrics(validation["y"], val_pred, validation["groups"]),
            "test": regression_metrics(test["y"], test_pred, test["groups"]),
            "seed_test":seeded_test,"selected_epochs": [f.selection["selected_epoch"] for f in fits]}, test_pred


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True); result = {"status": "development-only; successful HNEI cohort untouched", "datasets": {}}
    for dataset in ("nasa_alt", "calce"):
        ids, train, validation, test, elapsed_scale = prepare(dataset); screening = []
        for config in CONFIGS:
            row, _ = fit_config(config, train, validation, test, SELECTION_SEEDS, elapsed_scale); screening.append(row)
            print(dataset, config["name"], row["selected_gain"], row["validation"]["pooled"]["rmse"], row["test"]["pooled"]["r2"], flush=True)
        selected = min(screening, key=lambda x: (x["validation"]["pooled"]["rmse"], x["config"]["name"]))
        final, prediction = fit_config(selected["config"], train, validation, test, FINAL_SEEDS, elapsed_scale)
        # Fair same-feature direct-NN reference when the chosen PP uses the
        # explicit causal history basis.
        enhanced_plain = None
        if selected["config"].get("history_basis"):
            cfg={"history_basis":True,"name":"plain_history"}
            def augment(rows):
                x=np.asarray(rows["x"],dtype=np.float32);recent=x[:,2:5];elapsed=np.maximum(x[:,1],1e-4);global_rate=x[:,7]/elapsed
                extra=np.column_stack([np.median(recent,axis=1),np.std(recent,axis=1),np.min(recent,axis=1),np.max(recent,axis=1),global_rate,np.clip(np.median(recent,axis=1)/(global_rate+np.sign(global_rate)*1e-4+1e-4),-20,20)])
                return {**rows,"x":np.column_stack([x,extra]).astype(np.float32)}
            atr,ava,ate=map(augment,(train,validation,test));pred=[]
            for seed in FINAL_SEEDS: pred.append(predict_plain(fit_plain(atr,ava,seed=seed,max_epochs=450,patience=90),ate["x"]))
            enhanced_plain=regression_metrics(ate["y"],np.mean(pred,axis=0),ate["groups"])
        result["datasets"][dataset] = {"split_ids": ids, "selection_seeds": SELECTION_SEEDS, "screening": screening,
            "selected_config": selected["config"]["name"], "final_five_seed": final,"same_feature_plain":enhanced_plain}
        np.savez_compressed(OUT / f"{dataset}_predictions.npz", y=test["y"], groups=test["groups"], prediction=prediction)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {"selected": v["selected_config"], "gain": v["final_five_seed"]["selected_gain"], "r2": v["final_five_seed"]["test"]["pooled"]["r2"]} for k,v in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
