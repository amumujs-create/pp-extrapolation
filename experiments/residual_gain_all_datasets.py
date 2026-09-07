#!/usr/bin/env python3
"""Development benchmark for validation-calibrated PP residual confidence."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation import (
    LatentRegimeFit,
    LatentRegimePPNet,
    combine_residual_gain,
    latent_regime_components,
    regression_metrics,
    select_affine_initialization,
    select_group_robust_residual_gain,
    select_residual_gain,
)
from pp_extrapolation.model import _affine_prediction, fit_pp
from pp_extrapolation.support_gate import predict_components


SEEDS = (42, 43, 44, 45, 46)


def summary(y, groups, matrix):
    matrix = np.asarray(matrix)
    metrics = [regression_metrics(y, row, groups) for row in matrix]
    r2 = [float(row["pooled"]["r2"]) for row in metrics]
    return {
        "mean_r2": statistics.mean(r2),
        "sample_sd_r2": statistics.stdev(r2),
        "ensemble": regression_metrics(y, matrix.mean(axis=0), groups),
        "per_seed": metrics,
    }


def fit_predict_base(train, validation, test, max_epochs=300):
    affine_selection = select_affine_initialization(train, validation)
    base, calibrated, choices = [], [], []
    validation_affine, validation_residual = [], []
    test_affine, test_residual = [], []
    for seed in SEEDS:
        fit = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=affine_selection,
            max_epochs=max_epochs,
        )
        va, vr = predict_components(fit, validation["x"])
        validation_affine.append(va)
        validation_residual.append(vr)
        choice = select_residual_gain(
            va, vr, validation["y"], output_cap=fit.target_scale
        )
        ta, tr = predict_components(fit, test["x"])
        test_affine.append(ta)
        test_residual.append(tr)
        base.append(combine_residual_gain(ta, tr, gain=1.0, output_cap=fit.target_scale))
        calibrated.append(
            combine_residual_gain(
                ta, tr, gain=choice.gain, output_cap=fit.target_scale
            )
        )
        choices.append(
            {
                "seed": seed,
                "gain": choice.gain,
                "candidates": list(choice.candidates),
                "selected_epoch": fit.selection["selected_epoch"],
            }
        )
    global_choice = select_group_robust_residual_gain(
        np.mean(validation_affine, axis=0),
        np.mean(validation_residual, axis=0),
        validation["y"],
        validation["groups"],
        output_cap=affine_selection["target_scale"],
    )
    calibrated = [
        combine_residual_gain(
            affine,
            residual,
            gain=global_choice.gain,
            output_cap=affine_selection["target_scale"],
        )
        for affine, residual in zip(test_affine, test_residual)
    ]
    affine = _affine_prediction(
        affine_selection["initialization"],
        test["x"],
        affine_selection["center"],
        affine_selection["scale"],
        affine_selection["target_scale"],
    )
    return np.asarray(base), np.asarray(calibrated), {
        "global_gain": global_choice.gain,
        "global_candidates": list(global_choice.candidates),
        "per_seed_audit": choices,
    }, affine


def evaluate_base(train, validation, test, max_epochs=300):
    base, calibrated, choices, affine = fit_predict_base(
        train, validation, test, max_epochs=max_epochs
    )
    return {
        "n": {"train": len(train["y"]), "validation": len(validation["y"]), "test": len(test["y"])},
        "gain_choices": choices,
        "affine": regression_metrics(test["y"], affine, test["groups"]),
        "pp": summary(test["y"], test["groups"], base),
        "gain_pp": summary(test["y"], test["groups"], calibrated),
    }


def battery_rows(raw):
    from distance_uncertainty_pp import prepare_battery

    return prepare_battery(raw)


def evaluate_nasa(folds):
    truth, groups = [], []
    base_parts = [[] for _ in SEEDS]
    gain_parts = [[] for _ in SEEDS]
    affine_parts, choices = [], []
    for fold in folds:
        b, c, selected, affine = fit_predict_base(
            fold["train"], fold["validation"], fold["test"]
        )
        truth.append(fold["test"]["y"])
        groups.append(fold["test"]["groups"])
        affine_parts.append(affine)
        for index in range(len(SEEDS)):
            base_parts[index].append(b[index])
            gain_parts[index].append(c[index])
        choices.append({"test_cell": fold["test_cell"], **selected})
    y = np.concatenate(truth)
    g = np.concatenate(groups)
    base = np.asarray([np.concatenate(parts) for parts in base_parts])
    calibrated = np.asarray([np.concatenate(parts) for parts in gain_parts])
    return {
        "n": {"test": len(y), "folds": len(folds)},
        "gain_choices": choices,
        "affine": regression_metrics(y, np.concatenate(affine_parts), g),
        "pp": summary(y, g, base),
        "gain_pp": summary(y, g, calibrated),
    }


def evaluate_ncmapss(h5: Path, legacy: Path):
    from ncmapss_pp_benchmark import pp_features, rows

    sys.path.insert(0, str(legacy))
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_css import eval_all_bands_hard
    from ncmapss_tra_quantile_split import make_tra_hard_split

    split = make_tra_hard_split(h5, max_windows_per_unit=1500, random_seed=42)
    names = list(FEATURE_COLS)
    validation = rows(split.val, names)
    vx = validation["x"]
    all_x = pp_features(split.all_windows, names)
    base, calibrated, choices = [], [], []
    validation_affine, validation_residual = [], []
    test_affine, test_residual = [], []
    artifact = Path("results/ncmapss_pp_benchmark_v1")
    for seed in SEEDS:
        state = torch.load(artifact / f"seed{seed}.pt", map_location="cpu", weights_only=False)
        meta = json.loads((artifact / f"seed{seed}.json").read_text())["selected"]
        model = LatentRegimePPNet(
            len(state["center"]), meta["direction"], meta["knot"], width=24
        )
        model.load_state_dict(state["model_state"])
        fit = LatentRegimeFit(
            model, state["center"], state["scale"], state["target_scale"], meta
        )
        va, vr = latent_regime_components(fit, vx)
        validation_affine.append(va)
        validation_residual.append(vr)
        choice = select_residual_gain(
            va, vr, validation["y"], output_cap=fit.target_scale
        )
        ta, tr = latent_regime_components(fit, all_x)
        test_affine.append(ta)
        test_residual.append(tr)
        base.append(combine_residual_gain(ta, tr, gain=1.0, output_cap=fit.target_scale))
        calibrated.append(
            combine_residual_gain(ta, tr, gain=choice.gain, output_cap=fit.target_scale)
        )
        choices.append({"seed": seed, "gain": choice.gain, "candidates": list(choice.candidates)})

    global_choice = select_group_robust_residual_gain(
        np.mean(validation_affine, axis=0),
        np.mean(validation_residual, axis=0),
        validation["y"],
        validation["groups"],
        output_cap=fit.target_scale,
    )
    calibrated = [
        combine_residual_gain(
            affine, residual, gain=global_choice.gain, output_cap=fit.target_scale
        )
        for affine, residual in zip(test_affine, test_residual)
    ]

    def hard_summary(matrix):
        values = [eval_all_bands_hard(split, row)["hard_extrap"]["overall"] for row in matrix]
        r2 = [float(row["r2"]) for row in values]
        ensemble = eval_all_bands_hard(split, np.mean(matrix, axis=0))["hard_extrap"]["overall"]
        return {"mean_r2": statistics.mean(r2), "sample_sd_r2": statistics.stdev(r2), "ensemble": ensemble, "per_seed": values}

    return {
        "n": {"test": 159},
        "gain_choices": {
            "global_gain": global_choice.gain,
            "global_candidates": list(global_choice.candidates),
            "per_seed_audit": choices,
        },
        "pp": hard_summary(np.asarray(base)),
        "gain_pp": hard_summary(np.asarray(calibrated)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/residual_gain_all_v1"))
    parser.add_argument("--ncmapss-h5", type=Path, default=Path("data/N-CMAPSS_DS02-006.h5"))
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ca-css-ncmapss",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    legacy = args.legacy_root.resolve()
    sys.path.insert(0, str(legacy))
    from pae_boundary_realdata import prepare_dataset
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from run_affine_tail_external_three import prepare_hust, prepare_virkler

    torch.set_num_threads(2)
    started = time.time()
    datasets = {}

    for name, split in (("hust", prepare_hust()), ("virkler", prepare_virkler())):
        datasets[name] = evaluate_base(*split[:3])
        print(name, datasets[name]["pp"]["ensemble"]["pooled"]["r2"], datasets[name]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    folds, _ = prepare_folds()
    datasets["nasa"] = evaluate_nasa(folds)
    print("nasa", datasets["nasa"]["pp"]["ensemble"]["pooled"]["r2"], datasets["nasa"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    for name in ("sunwoda", "rwth", "mich"):
        raw, _ = prepare_dataset(name)
        datasets[name] = evaluate_base(*battery_rows(raw))
        print(name, datasets[name]["pp"]["ensemble"]["pooled"]["r2"], datasets[name]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    from extended_nn_benchmark import data as matr2019_data

    datasets["matr2019"] = evaluate_base(*matr2019_data())
    print("matr2019", datasets["matr2019"]["pp"]["ensemble"]["pooled"]["r2"], datasets["matr2019"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    from matr_batch2_confirmatory import endpoints, load_cells as load_batch2, make_rows as batch2_rows

    cells = load_batch2()
    cutoff = float(np.quantile(endpoints(cells, range(30)), 0.25))
    train0 = batch2_rows(cells, range(30), cutoff, train=True)
    boundary = float(train0["coordinate"].min())
    batch2 = (
        batch2_rows(cells, range(30), boundary, train=True),
        batch2_rows(cells, range(30, 39), boundary),
        batch2_rows(cells, range(39, 48), boundary),
    )
    datasets["matr_batch2"] = evaluate_base(*batch2)
    print("matr_batch2", datasets["matr_batch2"]["pp"]["ensemble"]["pooled"]["r2"], datasets["matr_batch2"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    from xjtu_untouched import build_cache as xjtu_cache, windows as xjtu_windows

    xraw = xjtu_cache()
    datasets["xjtu"] = evaluate_base(
        xjtu_windows(xraw, "37.5Hz11kN"),
        xjtu_windows(xraw, "35Hz12kN"),
        xjtu_windows(xraw, "40Hz10kN"),
    )
    print("xjtu", datasets["xjtu"]["pp"]["ensemble"]["pooled"]["r2"], datasets["xjtu"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    from femto_pp_prospective import split as femto_split
    from femto_bearing_loader import load_femto_phm2012

    femto_frame, femto_groups = load_femto_phm2012(root=Path("data/femto/raw"), file_stride=5)
    femto_endpoint = (
        femto_frame[femto_frame.unit.isin(femto_groups["test"])]
        .sort_values("cycle")
        .groupby("unit", as_index=False)
        .tail(1)
    )
    from femto_bearing_loader import FEATURE_COLS as FEMTO_FEATURES
    femto_test = {
        "x": femto_endpoint[FEMTO_FEATURES].to_numpy(np.float32),
        "y": femto_endpoint.RUL.to_numpy(np.float32),
        "groups": femto_endpoint.bearing.to_numpy(),
    }
    datasets["femto"] = evaluate_base(
        femto_split(femto_frame, femto_groups["train"]),
        femto_split(femto_frame, femto_groups["val"]),
        femto_test,
    )
    print("femto", datasets["femto"]["pp"]["ensemble"]["pooled"]["r2"], datasets["femto"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    from milling_locked_transfer import subset as milling_subset
    from nasa_milling_causal import prepare_causal_milling

    milling_raw, _ = prepare_causal_milling()
    milling_cut = float(np.quantile(milling_raw["train"]["health"], 0.60))
    datasets["milling"] = evaluate_base(
        milling_subset(milling_raw["train"], milling_raw["train"]["health"] <= milling_cut),
        milling_subset(milling_raw["validation"], milling_raw["validation"]["health"] > milling_cut),
        milling_subset(milling_raw["source"], milling_raw["source"]["health"] > milling_cut),
    )
    print("milling", datasets["milling"]["pp"]["ensemble"]["pooled"]["r2"], datasets["milling"]["gain_pp"]["ensemble"]["pooled"]["r2"], flush=True)

    datasets["ncmapss"] = evaluate_ncmapss(args.ncmapss_h5.resolve(), legacy)
    print("ncmapss", datasets["ncmapss"]["pp"]["ensemble"]["r2"], datasets["ncmapss"]["gain_pp"]["ensemble"]["r2"], flush=True)

    payload = {
        "experiment": "validation_calibrated_residual_gain_all_v1",
        "status": "post-hoc development on previously inspected datasets",
        "model": "affine prior + validation-calibrated scalar gain * PP neural residual",
        "gains": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "selection": "validation MSE only; no test-label selection",
        "seeds": list(SEEDS),
        "datasets": datasets,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
