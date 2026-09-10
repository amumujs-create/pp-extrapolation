#!/usr/bin/env python3
"""Matched v1.1 safety-gate ablation on the post-test Stanford development split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from pp_extrapolation import regression_metrics
from stanford_ppx_v11_safety import FINAL_SEEDS, fit_route, prepare

SOURCE = ROOT / "results/stanford_ppx_v11_safety_development/results.json"
OUT = ROOT / "results/stanford_ppx_v11_gate_ablation"


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite v1.1 gate ablation")
    archived = json.loads(SOURCE.read_text())
    best = min(archived["candidate_search"], key=lambda row: row["validation_rmse"])
    config = {key: best[key] for key in ("width", "learning_rate", "weight_decay")}
    train, validation, test, ids, hull = prepare()
    candidate_validation, candidate_test, epochs = fit_route(
        train, validation, test, config, best["trust"], FINAL_SEEDS,
    )
    gated = np.load(
        ROOT / "results/stanford_ppx_v11_safety_development/predictions.npz",
        allow_pickle=False,
    )
    y = gated["y"]
    groups = gated["groups"]
    gate_prediction = gated["selected"].mean(0)
    fallback_prediction = gated["fallback"].mean(0)
    if not np.array_equal(y, test["y"]) or not np.array_equal(groups, test["groups"]):
        raise RuntimeError("archived and reconstructed test rows are not aligned")
    payload = {
        "status": "post-test matched safety-gate ablation; not confirmation",
        "model": "PP-X v1.1",
        "split_ids": ids,
        "hull": hull,
        "arms": {
            "prior_always_on": {
                "selection": best,
                "final_epochs": epochs,
                "validation": regression_metrics(
                    validation["y"], candidate_validation.mean(0),
                    validation["groups"],
                ),
                "test": regression_metrics(y, candidate_test.mean(0), groups),
            },
            "v11_validation_gate": {
                "decision": archived["decision"],
                "test": regression_metrics(y, gate_prediction, groups),
            },
            "matched_mlp_fallback": {
                "test": regression_metrics(y, fallback_prediction, groups),
            },
        },
        "gate_equals_fallback_max_abs": float(np.max(np.abs(
            gate_prediction - fallback_prediction
        ))),
    }
    np.savez_compressed(
        OUT / "predictions.npz", y=y, groups=groups,
        prior_always_on=candidate_test, gate=gate_prediction,
        fallback=fallback_prediction,
    )
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        name: row["test"]["pooled"] for name, row in payload["arms"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
