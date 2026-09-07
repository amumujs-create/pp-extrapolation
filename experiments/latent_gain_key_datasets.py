#!/usr/bin/env python3
"""Residual-gain audit for the original latent PP comparison datasets."""
import json
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation import (
    LatentRegimeFit,
    LatentRegimePPNet,
    combine_residual_gain,
    fit_latent_regime_pp,
    latent_regime_components,
    regression_metrics,
    select_affine_initialization,
    select_group_robust_residual_gain,
    select_residual_gain,
)
from regime_spline_deep_future import splits
from extended_nn_benchmark import data as matr_data

SEEDS = (42, 43, 44, 45, 46)


def loaded_fit(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = state["selection"]
    model = LatentRegimePPNet(
        len(state["center"]), meta["direction"], meta["knot"], width=24
    )
    model.load_state_dict(state["model_state"])
    return LatentRegimeFit(
        model, state["center"], state["scale"], state["target_scale"], meta
    )


def evaluate(parts, fits):
    train, validation, test = parts
    va, vr, ta, tr = [], [], [], []
    for fit in fits:
        a, r = latent_regime_components(fit, validation["x"])
        va.append(a); vr.append(r)
        a, r = latent_regime_components(fit, test["x"])
        ta.append(a); tr.append(r)
    choice = select_group_robust_residual_gain(
        np.mean(va, axis=0), np.mean(vr, axis=0), validation["y"],
        validation["groups"],
        output_cap=fits[0].target_scale,
    )
    base = np.asarray([
        combine_residual_gain(a, r, gain=1.0, output_cap=fits[0].target_scale)
        for a, r in zip(ta, tr)
    ])
    gain = np.asarray([
        combine_residual_gain(a, r, gain=choice.gain, output_cap=fits[0].target_scale)
        for a, r in zip(ta, tr)
    ])
    def result(matrix):
        ms = [regression_metrics(test["y"], p, test["groups"]) for p in matrix]
        rs = np.asarray([m["pooled"]["r2"] for m in ms])
        return {"mean_r2": float(rs.mean()), "sample_sd_r2": float(rs.std(ddof=1)),
                "ensemble": regression_metrics(test["y"], matrix.mean(0), test["groups"])}
    rows = list(choice.candidates)
    base_mse = next(r["validation_mse"] for r in rows if r["gain"] == 1.0)
    return {"gain": choice.gain,
            "relative_validation_mse_gain": (base_mse-min(r["validation_mse"] for r in rows))/base_mse,
            "candidates": rows, "pp": result(base), "gain_pp": result(gain)}


def main():
    torch.set_num_threads(2)
    datasets = {}
    deep = splits()
    root = Path("results/original_latent_pp_replay_v1")
    for name in ("hust", "virkler"):
        fits = [loaded_fit(root / f"{name}_{seed}.pt") for seed in SEEDS]
        datasets[name] = evaluate(deep[name], fits)
        print(name, datasets[name]["gain"], datasets[name]["pp"]["ensemble"]["pooled"]["r2"], datasets[name]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    parts = matr_data()
    archived = json.load(open("results/matr_2019_latent_confirmatory/results.json"))
    affine = select_affine_initialization(parts[0], parts[1])
    fits = []
    for seed, selection in zip(SEEDS, archived["runs"]["latent"]):
        fits.append(fit_latent_regime_pp(
            parts[0], parts[1], seed=seed, affine_selection=affine, max_epochs=300,
            separation_weight=selection["separation"], gate_weight=selection["gate_weight"],
        ))
    datasets["matr2019"] = evaluate(parts, fits)
    print("matr2019", datasets["matr2019"]["gain"], datasets["matr2019"]["pp"]["ensemble"]["pooled"]["r2"], datasets["matr2019"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    out = Path("results/latent_gain_key_v1"); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "status": "post-hoc development; gain selected by validation ensemble only",
        "datasets": datasets,
    }, indent=2))


if __name__ == "__main__":
    main()
