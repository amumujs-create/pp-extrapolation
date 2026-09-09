#!/usr/bin/env python3
"""Thirty distinct validation candidates for every available PP split.

The historical 29-run audit is retained.  This runner is a new, explicitly
30-candidate protocol: 24 architecture settings at penalty .03, then six
non-overlapping penalty refinements around the selected architecture.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]
import extrapolation_competitors_matr as base
from extrapolation_competitors_all import datasets
from pp_extrapolation import regression_metrics, select_affine_initialization

OUT = ROOT / "results/final_30_candidate_competitors_v1"
SEEDS = (42, 43, 44, 45, 46)
ARCH = [(w, d, lr, wd) for w, d in ((16, 1), (32, 1), (32, 2), (64, 2))
        for lr in (2e-4, 5e-4, 1e-3) for wd in (.1, 2.)]
REFINE_PENALTIES = (3e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def tune(kind: str, parts: tuple) -> dict:
    affine = select_affine_initialization(parts[0], parts[1])
    search = []
    for arch in ARCH:
        cfg = (*arch, .03)
        info, _ = base.fit_neural(kind, cfg, 42, parts, affine)
        search.append({"stage": "architecture", "config": cfg, **info})
    selected_arch = min(search, key=lambda row: row["validation_mse"])["config"][:4]
    for penalty in REFINE_PENALTIES:
        cfg = (*selected_arch, penalty)
        info, _ = base.fit_neural(kind, cfg, 42, parts, affine)
        search.append({"stage": "penalty", "config": cfg, **info})
    chosen = min(search, key=lambda row: row["validation_mse"])["config"]
    predictions, runs = [], []
    for seed in SEEDS:
        info, prediction = base.fit_neural(kind, chosen, seed, parts, affine, True)
        predictions.append(prediction)
        runs.append({"seed": seed, **info,
                     "metrics": regression_metrics(parts[2]["y"], prediction, parts[2]["groups"])})
    artifact_dir = OUT / "predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_dir / f"{CURRENT_DATASET}_{kind}.npz", prediction=np.asarray(predictions),
                        y=parts[2]["y"], groups=parts[2]["groups"])
    return {"search_budget": len(search), "candidate_contract": "24 architecture + 6 distinct penalty configurations",
            "selected_config": chosen, "search": search, "runs": runs,
            "ensemble": regression_metrics(parts[2]["y"], np.mean(predictions, axis=0), parts[2]["groups"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="xjtu,femto")
    parser.add_argument("--force", action="store_true", help="recreate requested prediction artifacts")
    args = parser.parse_args()
    wanted = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {
        "protocol": "30 distinct validation candidates; five seed refits; no test-label selection", "datasets": {}}
    available = datasets()
    if "ncmapss" in wanted:
        from apps.ncmapss_data_utils import FEATURE_COLS
        from ncmapss_pp_benchmark import rows
        from ncmapss_tra_quantile_split import make_tra_hard_split
        split = make_tra_hard_split(ROOT / "data/N-CMAPSS_DS02-006.h5", max_windows_per_unit=1500, random_seed=42)
        names = list(FEATURE_COLS)
        available["ncmapss"] = (rows(split.train, names), rows(split.val, names), rows(split.test, names))
    for name in wanted:
        if name not in available:
            raise KeyError(f"unknown or unavailable dataset {name}")
        parts = available[name]
        global CURRENT_DATASET
        CURRENT_DATASET = name
        if args.force:
            result["datasets"].pop(name, None)
        result["datasets"].setdefault(name, {"validation_groups": int(len(np.unique(parts[1]["groups"])))})
        for kind in ("vrex", "groupdro", "monotone"):
            if kind in result["datasets"][name]:
                continue
            started = time.time()
            result["datasets"][name][kind] = tune(kind, parts)
            result["datasets"][name][kind]["seconds"] = time.time() - started
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(name, kind, result["datasets"][name][kind]["ensemble"]["pooled"]["r2"], flush=True)


if __name__ == "__main__":
    main()
