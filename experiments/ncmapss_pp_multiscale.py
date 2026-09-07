#!/usr/bin/env python3
"""Validation-selected causal multiscale PP on the N-CMAPSS hard split."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation import fit_latent_regime_pp, predict_latent_regime, select_affine_initialization


PRESETS = ("basic", "moments", "multiscale")
SEPARATION_GRID = (0.0, 0.001, 0.01)


def make_features(batch, feature_names: list[str], preset: str) -> np.ndarray:
    seq = np.asarray(batch.X_seq, dtype=np.float32)
    tra = feature_names.index("TRA")
    last = seq[:, -1]
    blocks = [last[:, tra : tra + 1], np.delete(last, tra, axis=1)]
    if preset == "basic":
        blocks += [seq.mean(axis=1), (seq[:, -1] - seq[:, 0]) / max(seq.shape[1] - 1, 1)]
    elif preset == "moments":
        blocks += [
            seq.mean(axis=1),
            seq.std(axis=1),
            np.median(seq, axis=1),
            (seq[:, -1] - seq[:, 0]) / max(seq.shape[1] - 1, 1),
        ]
    elif preset == "multiscale":
        for horizon in sorted({min(5, seq.shape[1]), min(10, seq.shape[1]), seq.shape[1]}):
            tail = seq[:, -horizon:]
            blocks += [
                tail.mean(axis=1),
                tail.std(axis=1),
                (tail[:, -1] - tail[:, 0]) / max(horizon - 1, 1),
            ]
    else:
        raise ValueError(preset)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def as_rows(batch, names: list[str], preset: str) -> dict:
    return {
        "x": make_features(batch, names, preset),
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
        "--output", type=Path, default=Path("results/ncmapss_pp_multiscale_v1")
    )
    parser.add_argument("--fixed-preset", choices=PRESETS)
    parser.add_argument("--fixed-separation", type=float)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy_root.expanduser().resolve()))
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_css import eval_all_bands_hard
    from ncmapss_tra_quantile_split import make_tra_hard_split

    torch.set_num_threads(2)
    split = make_tra_hard_split(args.h5.expanduser().resolve(), max_windows_per_unit=1500, random_seed=42)
    names = list(FEATURE_COLS)
    prepared = {}
    for preset in PRESETS:
        train = as_rows(split.train, names, preset)
        validation = as_rows(split.val, names, preset)
        prepared[preset] = (train, validation, select_affine_initialization(train, validation))

    args.output.mkdir(parents=True, exist_ok=True)
    runs, predictions = [], []
    for seed in range(42, 47):
        candidates = []
        best = None
        preset_grid = (args.fixed_preset,) if args.fixed_preset else PRESETS
        separation_grid = (
            (args.fixed_separation,)
            if args.fixed_separation is not None
            else SEPARATION_GRID
        )
        for preset in preset_grid:
            train, validation, affine = prepared[preset]
            for separation in separation_grid:
                fit = fit_latent_regime_pp(
                    train,
                    validation,
                    seed=seed,
                    affine_selection=affine,
                    max_epochs=300,
                    patience=70,
                    separation_weight=separation,
                    gate_weight=0.0,
                )
                candidate = {
                    "preset": preset,
                    "separation_weight": separation,
                    **fit.selection,
                }
                candidates.append(candidate)
                if best is None or candidate["validation_mse"] < best[0]:
                    best = (candidate["validation_mse"], preset, fit, candidate)
        assert best is not None
        preset, fit, selected = best[1], best[2], best[3]
        prediction = predict_latent_regime(fit, make_features(split.all_windows, names, preset))
        metrics = eval_all_bands_hard(split, prediction)["hard_extrap"]["overall"]
        runs.append({"seed": seed, "selected": selected, "candidates": candidates, "raw": metrics})
        predictions.append(prediction)
        np.savez_compressed(
            args.output / f"seed{seed}.npz",
            prediction=prediction,
            y=np.asarray(split.all_windows.y),
            units=np.asarray(split.all_windows.units),
            cycles=np.asarray(split.all_windows.cycles_end),
        )
        print(
            f"seed={seed} preset={preset} val={selected['validation_mse']:.4f} test_R2={metrics['r2']:.6f}",
            flush=True,
        )

    ensemble_prediction = np.mean(predictions, axis=0)
    ensemble = eval_all_bands_hard(split, ensemble_prediction)["hard_extrap"]["overall"]
    values = [float(run["raw"]["r2"]) for run in runs]
    result = {
        "status": "post-hoc PP development; feature preset selected only by validation MSE",
        "protocol": {
            "split": "unseen engine x high TRA x late life",
            "presets": list(preset_grid),
            "separation_grid": list(separation_grid),
            "test_used_for_selection": False,
        },
        "n_test": int(ensemble["n"]),
        "runs": runs,
        "single_seed_r2_mean": statistics.mean(values),
        "single_seed_r2_sample_sd": statistics.stdev(values),
        "ensemble": ensemble,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ensemble R2={ensemble['r2']:.6f}", flush=True)


if __name__ == "__main__":
    main()
