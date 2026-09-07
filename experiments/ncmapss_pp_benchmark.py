#!/usr/bin/env python3
"""Evaluate latent PP on the archived N-CMAPSS hard protocol.

The comparison sequence models use a length-T history.  PP receives a causal
tabular summary of exactly that history: last value, window mean, and endpoint
slope.  The extrapolation coordinate is the last observed TRA value and is
placed in column zero, as required by the latent PP gate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation import (
    fit_latent_regime_pp,
    predict_latent_regime,
    select_affine_initialization,
)


REGULARIZER_GRID = tuple(
    (separation, gate)
    for separation in (0.0, 0.001, 0.01)
    for gate in (0.0, 0.001, 0.01)
)


def pp_features(batch, feature_names: list[str]) -> np.ndarray:
    sequence = np.asarray(batch.X_seq, dtype=np.float32)
    if sequence.ndim != 3:
        raise ValueError(f"expected [rows, time, features], got {sequence.shape}")
    try:
        tra_index = feature_names.index("TRA")
    except ValueError as error:
        raise ValueError("TRA must be present in N-CMAPSS sequence features") from error
    last = sequence[:, -1]
    mean = sequence.mean(axis=1)
    slope = (sequence[:, -1] - sequence[:, 0]) / max(sequence.shape[1] - 1, 1)
    context_last = np.delete(last, tra_index, axis=1)
    return np.column_stack((last[:, tra_index], context_last, mean, slope)).astype(
        np.float32
    )


def rows(batch, feature_names: list[str]) -> dict:
    return {
        "x": pp_features(batch, feature_names),
        "y": np.asarray(batch.y, dtype=np.float32),
        "groups": np.asarray(batch.units, dtype=str),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ca-css-ncmapss",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ncmapss_pp_benchmark_v1"),
    )
    parser.add_argument("--max-windows-per-unit", type=int, default=1500)
    args = parser.parse_args()
    h5 = args.h5.expanduser().resolve()
    if not h5.is_file():
        raise FileNotFoundError(f"N-CMAPSS H5 not found: {h5}")

    legacy = args.legacy_root.expanduser().resolve()
    sys.path.insert(0, str(legacy))
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_css import eval_all_bands_hard
    from ncmapss_tra_quantile_split import make_tra_hard_split
    from ncmapss_unit_mono import apply_unit_isotonic_post

    torch.set_num_threads(2)
    split = make_tra_hard_split(
        h5, max_windows_per_unit=args.max_windows_per_unit, random_seed=42
    )
    train = rows(split.train, list(FEATURE_COLS))
    validation = rows(split.val, list(FEATURE_COLS))
    all_x = pp_features(split.all_windows, list(FEATURE_COLS))
    affine = select_affine_initialization(train, validation)
    args.output.mkdir(parents=True, exist_ok=True)

    runs = []
    raw_predictions = []
    predictions = []
    for seed in range(42, 47):
        candidates = []
        selected = None
        for separation, gate in REGULARIZER_GRID:
            fit = fit_latent_regime_pp(
                train,
                validation,
                seed=seed,
                affine_selection=affine,
                max_epochs=300,
                patience=70,
                separation_weight=separation,
                gate_weight=gate,
            )
            candidate = {
                "separation_weight": separation,
                "gate_weight": gate,
                **fit.selection,
            }
            candidates.append(candidate)
            if selected is None or candidate["validation_mse"] < selected[0]:
                selected = (candidate["validation_mse"], fit, candidate)
        assert selected is not None

        fit = selected[1]
        raw_prediction = predict_latent_regime(fit, all_x)
        iso_prediction = apply_unit_isotonic_post(
            raw_prediction,
            split.all_windows.units,
            split.all_windows.cycles_end,
        )
        raw_metrics = eval_all_bands_hard(split, raw_prediction)["hard_extrap"]["overall"]
        iso_metrics = eval_all_bands_hard(split, iso_prediction)["hard_extrap"]["overall"]
        record = {
            "seed": seed,
            "selected": selected[2],
            "candidates": candidates,
            "raw": raw_metrics,
            "isotonic": iso_metrics,
        }
        runs.append(record)
        raw_predictions.append(raw_prediction)
        predictions.append(iso_prediction)
        np.savez_compressed(
            args.output / f"seed{seed}.npz",
            raw_prediction=raw_prediction,
            isotonic_prediction=iso_prediction,
            y=np.asarray(split.all_windows.y),
            units=np.asarray(split.all_windows.units),
            cycles=np.asarray(split.all_windows.cycles_end),
        )
        torch.save(
            {
                "model_state": fit.model.state_dict(),
                "center": fit.center,
                "scale": fit.scale,
                "target_scale": fit.target_scale,
                "selection": fit.selection,
            },
            args.output / f"seed{seed}.pt",
        )
        (args.output / f"seed{seed}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print(f"seed={seed} PP+iso R2={iso_metrics['r2']:.6f}", flush=True)

    raw_ensemble_prediction = np.mean(raw_predictions, axis=0)
    raw_ensemble = eval_all_bands_hard(split, raw_ensemble_prediction)["hard_extrap"]["overall"]
    ensemble_prediction = np.mean(predictions, axis=0)
    ensemble = eval_all_bands_hard(split, ensemble_prediction)["hard_extrap"]["overall"]
    raw_r2_values = [float(run["raw"]["r2"]) for run in runs]
    r2_values = np.asarray([run["isotonic"]["r2"] for run in runs])
    result = {
        "status": "post-hoc PP development on previously inspected N-CMAPSS hard cohort",
        "protocol": {
            "split": "unseen engine x high TRA x late life",
            "max_windows_per_unit": args.max_windows_per_unit,
            "features": "last + window mean + endpoint slope; TRA last is gate coordinate",
            "selection": "nine regularizer candidates per seed by validation MSE",
            "epochs": 300,
            "patience": 70,
            "postprocessing": "same per-unit isotonic operation used by sequence baselines",
        },
        "n_test": int(ensemble["n"]),
        "runs": runs,
        "raw_single_seed_r2_mean": statistics.mean(raw_r2_values),
        "raw_single_seed_r2_sample_sd": statistics.stdev(raw_r2_values),
        "raw_ensemble": raw_ensemble,
        "isotonic_single_seed_r2_mean": float(r2_values.mean()),
        "isotonic_single_seed_r2_sample_sd": statistics.stdev(r2_values.tolist()),
        "isotonic_ensemble": ensemble,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"ensemble raw PP R2={raw_ensemble['r2']:.6f}", flush=True)
    print(f"ensemble PP+iso R2={ensemble['r2']:.6f}", flush=True)


if __name__ == "__main__":
    main()
