#!/usr/bin/env python3
"""Nested leave-one-protocol-out stability audit for the HUST PP head."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evidence_gated_modules import ARMS, fit_arm_predictions
from pp_extrapolation.metrics import regression_metrics


FEATURES = ["capacity_ah", "cycle", "prefix_rate", "recent_rate"]


def mean_mse(rows: list[dict], truth: np.ndarray, arm: str) -> float:
    return float(np.mean([
        np.mean((row["test_prediction"][arm] - truth) ** 2) for row in rows
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output", default="results/hust_protocol_stability_v1")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.legacy_root).expanduser().resolve()))
    from run_affine_tail_external_three import (
        causal_hust_frame,
        evenly_spaced_per_group,
        frame_to_arrays,
    )

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    torch.set_num_threads(2)
    frame, _ = causal_hust_frame()
    source = frame[frame.protocol <= 6].copy()
    folds = []
    for protocol in range(1, 7):
        train_frame = source[
            (source.protocol != protocol) & (source.capacity_ah >= 1.05)
        ].copy()
        train_frame = evenly_spaced_per_group(train_frame, 250)
        train_min = float(train_frame.capacity_ah.min())
        heldout = source[
            (source.protocol == protocol) & (source.capacity_ah < train_min)
        ].copy()
        cells = sorted(heldout.unit.unique().tolist())
        split = max(1, len(cells) // 2)
        checkpoint_cells, audit_cells = cells[:split], cells[split:]
        checkpoint_frame = evenly_spaced_per_group(
            heldout[heldout.unit.isin(checkpoint_cells)].copy(), 160
        )
        audit_frame = heldout[heldout.unit.isin(audit_cells)].copy()
        train = frame_to_arrays(train_frame, FEATURES)
        checkpoint = frame_to_arrays(checkpoint_frame, FEATURES)
        audit = frame_to_arrays(audit_frame, FEATURES)
        rows = fit_arm_predictions(train, checkpoint, audit, seeds)
        arm_mse = {arm: mean_mse(rows, audit["y"], arm) for arm in ARMS}
        arm_r2 = {}
        for arm in ARMS:
            values = [
                regression_metrics(
                    audit["y"], row["test_prediction"][arm], audit["groups"]
                )["pooled"]["r2"]
                for row in rows
            ]
            arm_r2[arm] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=0)),
            }
        pp_seed_wins = sum(
            np.mean((row["test_prediction"]["pp"] - audit["y"]) ** 2)
            < np.mean((row["test_prediction"]["plain_nn"] - audit["y"]) ** 2)
            for row in rows
        )
        pp_relative_gain = (
            arm_mse["plain_nn"] - arm_mse["pp"]
        ) / max(arm_mse["plain_nn"], 1e-12)
        fold = {
            "heldout_protocol": protocol,
            "checkpoint_cells": checkpoint_cells,
            "audit_cells": audit_cells,
            "n_train": len(train["y"]),
            "n_checkpoint": len(checkpoint["y"]),
            "n_audit": len(audit["y"]),
            "train_capacity_min": train_min,
            "arm_audit_mse": arm_mse,
            "arm_audit_r2": arm_r2,
            "pp_relative_mse_gain_vs_plain": float(pp_relative_gain),
            "pp_seed_wins_vs_plain": int(pp_seed_wins),
            "pp_stable": bool(pp_relative_gain >= 0.02 and pp_seed_wins >= 4),
        }
        folds.append(fold)
        print(
            f"protocol={protocol} plain={arm_r2['plain_nn']['mean']:.4f} "
            f"PP={arm_r2['pp']['mean']:.4f} wins={pp_seed_wins}/5 "
            f"stable={fold['pp_stable']}",
            flush=True,
        )
    stable_count = sum(row["pp_stable"] for row in folds)
    payload = {
        "experiment": "hust_protocol_stability_v1",
        "status": "retrospective nested source-protocol audit",
        "seeds": list(seeds),
        "protocols": folds,
        "summary": {
            "pp_stable_protocols": int(stable_count),
            "total_protocols": len(folds),
            "pp_route_pass_5_of_6": bool(stable_count >= 5),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# HUST source-protocol stability audit", "",
        "Each protocol 1–6 is held out in turn. Half of its cells select checkpoints; "
        "the remaining cells audit the route. All audit rows lie below the inner-train "
        "capacity support. PP passes a protocol only with at least 2% audit-MSE gain "
        "and at least four of five seed wins over plain NN.", "",
        "| protocol | plain NN R² | PP R² | PP MSE gain | seed wins | stable |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['heldout_protocol']} | "
            f"{row['arm_audit_r2']['plain_nn']['mean']:.3f}±"
            f"{row['arm_audit_r2']['plain_nn']['sd']:.3f} | "
            f"{row['arm_audit_r2']['pp']['mean']:.3f}±"
            f"{row['arm_audit_r2']['pp']['sd']:.3f} | "
            f"{row['pp_relative_mse_gain_vs_plain']:+.3f} | "
            f"{row['pp_seed_wins_vs_plain']}/5 | {row['pp_stable']} |"
        )
    lines += [
        "", f"PP route stability: **{stable_count}/6 protocols**. "
        f"Five-of-six gate: **{stable_count >= 5}**.", "",
        "This audit was designed after the outer HUST result was observed and is "
        "development evidence only.", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
