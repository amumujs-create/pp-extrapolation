#!/usr/bin/env python3
"""Nested unseen-specimen stability audit for the Virkler PP head."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evidence_gated_modules import ARMS, fit_arm_predictions
from pp_extrapolation.metrics import regression_metrics


FEATURES = [
    "crack_length_mm",
    "elapsed_kcycles",
    "prefix_rate_mm_per_kcycle",
    "recent_rate_mm_per_kcycle",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output", default="results/virkler_unit_stability_v1")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.legacy_root).expanduser().resolve()))
    from run_affine_tail_external_three import (
        frame_to_arrays,
        prepare_virkler,
        reconstruct_virkler,
    )

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    torch.set_num_threads(2)
    frame, _, _ = reconstruct_virkler()
    _, _, _, meta = prepare_virkler()
    source_units = np.asarray(meta["audit"]["train_units"], dtype=np.int64)
    folds = []
    for fold_index in range(3):
        heldout_units = source_units[np.arange(len(source_units)) % 3 == fold_index]
        fit_units = source_units[np.arange(len(source_units)) % 3 != fold_index]
        midpoint = len(heldout_units) // 2
        checkpoint_units = heldout_units[:midpoint]
        audit_units = heldout_units[midpoint:]
        train_frame = frame[
            frame.unit.isin(fit_units) & (frame.crack_length_mm <= 33)
        ].copy()
        checkpoint_frame = frame[
            frame.unit.isin(checkpoint_units) & (frame.crack_length_mm > 33)
        ].copy()
        audit_frame = frame[
            frame.unit.isin(audit_units) & (frame.crack_length_mm > 33)
        ].copy()
        train = frame_to_arrays(train_frame, FEATURES)
        checkpoint = frame_to_arrays(checkpoint_frame, FEATURES)
        audit = frame_to_arrays(audit_frame, FEATURES)
        rows = fit_arm_predictions(train, checkpoint, audit, seeds)
        arm_mse, arm_r2 = {}, {}
        for arm in ARMS:
            seed_mse, seed_r2 = [], []
            for row in rows:
                prediction = row["test_prediction"][arm]
                seed_mse.append(float(np.mean((prediction - audit["y"]) ** 2)))
                seed_r2.append(regression_metrics(
                    audit["y"], prediction, audit["groups"]
                )["pooled"]["r2"])
            arm_mse[arm] = float(np.mean(seed_mse))
            arm_r2[arm] = {
                "mean": float(np.mean(seed_r2)),
                "sd": float(np.std(seed_r2, ddof=0)),
            }
        gain = (arm_mse["plain_nn"] - arm_mse["pp"]) / max(
            arm_mse["plain_nn"], 1e-12
        )
        wins = sum(
            np.mean((row["test_prediction"]["pp"] - audit["y"]) ** 2)
            < np.mean((row["test_prediction"]["plain_nn"] - audit["y"]) ** 2)
            for row in rows
        )
        stable = gain >= 0.02 and wins >= 4
        folds.append({
            "fold": fold_index,
            "fit_units": fit_units.tolist(),
            "checkpoint_units": checkpoint_units.tolist(),
            "audit_units": audit_units.tolist(),
            "arm_audit_mse": arm_mse,
            "arm_audit_r2": arm_r2,
            "pp_relative_mse_gain_vs_plain": float(gain),
            "pp_seed_wins_vs_plain": int(wins),
            "pp_stable": bool(stable),
        })
        print(
            f"fold={fold_index} plain={arm_r2['plain_nn']['mean']:.4f} "
            f"PP={arm_r2['pp']['mean']:.4f} wins={wins}/5 stable={stable}",
            flush=True,
        )
    stable_count = sum(row["pp_stable"] for row in folds)
    payload = {
        "experiment": "virkler_unit_stability_v1",
        "status": "retrospective nested source-unit audit",
        "seeds": list(seeds),
        "folds": folds,
        "summary": {
            "pp_stable_folds": int(stable_count),
            "total_folds": len(folds),
            "unanimous_pp_route": bool(stable_count == len(folds)),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Virkler source-unit stability audit", "",
        "The 48 source specimens are divided into three folds. Each held-out fold is "
        "split again into checkpoint and route-audit specimens. Training uses crack "
        "length <=33 mm and both held-out subsets use the >33 mm tail.", "",
        "| fold | plain NN R² | PP R² | PP MSE gain | seed wins | stable |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['fold']} | {row['arm_audit_r2']['plain_nn']['mean']:.3f}±"
            f"{row['arm_audit_r2']['plain_nn']['sd']:.3f} | "
            f"{row['arm_audit_r2']['pp']['mean']:.3f}±"
            f"{row['arm_audit_r2']['pp']['sd']:.3f} | "
            f"{row['pp_relative_mse_gain_vs_plain']:+.3f} | "
            f"{row['pp_seed_wins_vs_plain']}/5 | {row['pp_stable']} |"
        )
    lines += [
        "", f"Unanimous PP route: **{stable_count == len(folds)}** "
        f"({stable_count}/{len(folds)} folds).", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
