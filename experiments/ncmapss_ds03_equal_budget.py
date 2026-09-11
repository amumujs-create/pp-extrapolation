#!/usr/bin/env python3
"""Frozen post-reveal equal-budget comparators for prospective DS03."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from full_equal_candidate_budget import MODELS, atomic_json, run_one
from pp_extrapolation.ds03_prospective import (
    TRAIN_UNITS,
    VALIDATION_UNITS,
    causal_features,
    load_development,
    load_revealed_test,
)

OUT = ROOT / "results/ncmapss_ds03_equal_budget_v1"


def parts(h5_path: Path):
    development = causal_features(load_development(h5_path), "direct")
    test = causal_features(load_revealed_test(h5_path), "direct")
    train_mask = np.isin(development["groups"], np.asarray(TRAIN_UNITS, str))
    validation_mask = np.isin(
        development["groups"], np.asarray(VALIDATION_UNITS, str)
    )

    def take(rows, mask):
        return {
            "x": rows["x"][mask],
            "y": rows["y"][mask],
            "groups": rows["groups"][mask],
        }

    return (
        take(development, train_mask),
        take(development, validation_mask),
        take(test, np.ones(len(test["y"]), dtype=bool)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", type=Path, default=ROOT / "data/N-CMAPSS_DS03-012.h5"
    )
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    split = parts(args.h5)
    result_path = args.out / "results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "status": "post-reveal comparator execution under preregistered budget",
        "candidate_budget": 30,
        "search_seed": 42,
        "refit_seeds": [42, 43, 44, 45, 46],
        "models": {},
    }
    for model in (name for name in args.models.split(",") if name):
        if model in result["models"]:
            continue
        result["models"][model] = run_one(
            "ncmapss_ds03", model, split, args.out / model
        )
        atomic_json(result_path, result)
        print("DONE", model, result["models"][model]["ensemble"]["pooled"]["r2"])


if __name__ == "__main__":
    main()
