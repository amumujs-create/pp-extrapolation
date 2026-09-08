#!/usr/bin/env python3
"""Frozen cross-chemistry confirmation of normalized Boundary-Quotient PP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from naion_prefix_gate_eval import features_at
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import (
    fit_boundary_quotient_pp,
    predict_boundary_affine,
    predict_boundary_quotient,
    regression_metrics,
)

DATA = ROOT / "data" / "znion_bq_confirmatory"
OUT = ROOT / "results" / "znion_bq_confirmatory_v1"
SEEDS = (42, 43, 44, 45, 46)
CONFIG = dict(width=64, alpha=10.0, learning_rate=1e-3,
              weight_decay=1e-2, residual_bound=0.5,
              max_epochs=300, patience=50)
EXPECTED = {"train": 6, "validation": 3, "test": 3}


def read_cell(path: Path) -> dict | None:
    frame = pd.read_excel(path, sheet_name="循环", usecols=["循环序号", "放电容量/mAh"])
    frame.columns = ["cycle", "capacity"]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[(frame.cycle >= 10) & (frame.capacity > 0)].sort_values("cycle")
    frame = frame.groupby("cycle", as_index=False).capacity.max()
    nominal_rows = frame.loc[frame.cycle == 10, "capacity"]
    if nominal_rows.empty:
        return None
    nominal = float(nominal_rows.iloc[0])
    frame["soh"] = frame.capacity / nominal
    crossings = np.flatnonzero(frame.soh.to_numpy() <= 0.8)
    if len(crossings) == 0:
        return None  # frozen rule: exclude right-censored cells; never extrapolate the label
    end = int(crossings[0])
    frame = frame.iloc[: end + 1].copy()
    # BatteryLife drops formation cycles and resets the remaining cycle index.
    return {"cycle": np.arange(1, len(frame) + 1, dtype=float),
            "capacity": frame.soh.to_numpy(float), "nominal_mAh": nominal}


def load_split(split: str, require_count: bool = True) -> tuple[dict, list[str]]:
    paths = sorted((DATA / split).glob("*.xlsx"))
    if require_count and len(paths) != EXPECTED[split]:
        raise RuntimeError(f"{split}: expected {EXPECTED[split]} locked files, found {len(paths)}")
    cells, excluded = {}, []
    for path in paths:
        cell = read_cell(path)
        if cell is None:
            excluded.append(path.name)
        else:
            cells[path.stem] = cell
    return cells, excluded


def make_rows(cells: dict, boundary: float, time_scale: float, side: str) -> dict:
    xx, yy, groups, margins = [], [], [], []
    for uid, cell in sorted(cells.items()):
        t, soh = cell["cycle"], cell["capacity"]
        span = t - t[0]
        mask = span <= boundary if side == "prefix" else span > boundary
        for i in np.flatnonzero(mask & (np.arange(len(t)) >= 3)):
            xx.append(features_at(t, soh, i, time_scale))
            yy.append(t[-1] - t[i])
            groups.append(uid)
            margins.append(max(float(soh[i]) - 0.8, 0.0))
    groups = np.asarray(groups)
    return {"x": np.asarray(xx, np.float32), "y": np.asarray(yy, np.float32),
            "groups": groups, "units": groups,
            "dataset": np.asarray(["znion"] * len(groups)),
            "margin": np.asarray(margins, np.float32)}


def evaluate(train: dict, validation: dict, evaluation: dict) -> dict:
    predictions = {"plain_mlp": [], "boundary_affine": [], "bq_pp": []}
    runs = []
    for seed in SEEDS:
        plain = fit_plain(train, validation, seed=seed, max_epochs=300, patience=50)
        bq = fit_boundary_quotient_pp(train, validation, seed=seed, **CONFIG)
        current = {
            "plain_mlp": predict_plain(plain, evaluation["x"]),
            "boundary_affine": predict_boundary_affine(bq, evaluation),
            "bq_pp": predict_boundary_quotient(bq, evaluation),
        }
        for name, value in current.items():
            predictions[name].append(value)
        runs.append({"seed": seed, "plain_epoch": plain["selected_epoch"],
                     "bq_epoch": bq.selection["selected_epoch"],
                     **{name: regression_metrics(evaluation["y"], pred, evaluation["groups"])
                        for name, pred in current.items()}})
    stacked = {key: np.asarray(value) for key, value in predictions.items()}
    ensemble = {name: regression_metrics(evaluation["y"], value.mean(0), evaluation["groups"])
                for name, value in stacked.items()}
    return {"runs": runs, "ensemble": ensemble}, stacked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train_cells, train_excluded = load_split("train")
    val_cells, val_excluded = load_split("validation")
    life = [cell["cycle"][-1] - cell["cycle"][0] for cell in train_cells.values()]
    boundary = float(np.floor(0.60 * np.median(life)))
    time_scale = float(max(life))
    train = make_rows(train_cells, boundary, time_scale, "prefix")
    validation = make_rows(val_cells, boundary, time_scale, "tail")
    if args.phase == "validation":
        evaluation, excluded = validation, val_excluded
    else:
        test_cells, excluded = load_split("test")
        evaluation = make_rows(test_cells, boundary, time_scale, "tail")
    scores, predictions = evaluate(train, validation, evaluation)
    result = {
        "status": "frozen pre-test validation" if args.phase == "validation" else "one-shot untouched confirmation",
        "phase": args.phase, "eol": "first observed SOH <= 0.80 after formation",
        "model": "RUL=max(SOH-0.80,0)*softplus(frozen affine + bounded NN residual)",
        "config": CONFIG, "seeds": SEEDS, "boundary": boundary,
        "time_scale": time_scale,
        "n": {"train_units": len(train_cells), "validation_units": len(val_cells),
              "evaluation_units": len(np.unique(evaluation["groups"])),
              "evaluation_rows": len(evaluation["y"])},
        "excluded_right_censored": {"train": train_excluded,
                                     "validation": val_excluded,
                                     args.phase: excluded},
        **scores,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.phase}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / f"{args.phase}_predictions.npz", y=evaluation["y"],
                        groups=evaluation["groups"], **predictions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
