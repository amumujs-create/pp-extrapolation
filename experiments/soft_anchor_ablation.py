#!/usr/bin/env python3
"""Validation-selected soft prior anchoring for PP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.model import fit_pp, predict, select_affine_initialization


def evaluate(train, validation, test, seeds, anchors, decays):
    affine = select_affine_initialization(train, validation)
    rows = []
    for seed in seeds:
        candidates = []
        for anchor in anchors:
            for decay in decays:
                fit = fit_pp(train, validation, seed=seed, affine_selection=affine,
                             affine_anchor_weight=anchor, residual_decay=decay)
                candidates.append((fit.selection["best_validation_mse"], anchor, decay, fit))
        _, selected_anchor, selected_decay, selected_fit = min(candidates, key=lambda z: (z[0], -z[1], z[2]))
        frozen = fit_pp(train, validation, seed=seed, affine_selection=affine)
        soft_metrics = regression_metrics(test["y"], predict(selected_fit, test["x"]), test["groups"])
        frozen_metrics = regression_metrics(test["y"], predict(frozen, test["x"]), test["groups"])
        rows.append({
            "seed": seed, "selected_anchor": selected_anchor, "selected_decay": selected_decay,
            "candidate_validation_mse": {f"anchor={a},decay={d}": float(v) for v, a, d, _ in candidates},
            "soft_anchor": soft_metrics, "frozen_pp": frozen_metrics,
        })
        print(f"seed={seed} anchor={selected_anchor:g} decay={selected_decay:g} soft={soft_metrics['pooled']['r2']:.4f} frozen={frozen_metrics['pooled']['r2']:.4f}", flush=True)
    def summary(key):
        v = np.array([r[key]["pooled"]["r2"] for r in rows])
        return {"pooled_r2_mean": float(v.mean()), "pooled_r2_sd": float(v.std())}
    return {"soft_anchor": summary("soft_anchor"), "frozen_pp": summary("frozen_pp"), "runs": rows}


def evaluate_folds(folds, seeds, anchors, decays):
    rows = []
    for seed in seeds:
        soft_parts, frozen_parts, chosen = [], [], []
        for fold in folds:
            affine = select_affine_initialization(fold["train"], fold["validation"])
            candidates = []
            for anchor in anchors:
                for decay in decays:
                    fit = fit_pp(fold["train"], fold["validation"], seed=seed,
                                 affine_selection=affine, affine_anchor_weight=anchor,
                                 residual_decay=decay)
                    candidates.append((fit.selection["best_validation_mse"], anchor, decay, fit))
            _, anchor, decay, fit = min(candidates, key=lambda z: (z[0], -z[1], z[2]))
            frozen = fit_pp(fold["train"], fold["validation"], seed=seed,
                            affine_selection=affine)
            soft_parts.append(predict(fit, fold["test"]["x"]))
            frozen_parts.append(predict(frozen, fold["test"]["x"]))
            chosen.append({"test_cell": fold["test_cell"], "anchor": anchor, "decay": decay})
        truth = np.concatenate([f["test"]["y"] for f in folds])
        groups = np.concatenate([f["test"]["groups"] for f in folds])
        soft = regression_metrics(truth, np.concatenate(soft_parts), groups)
        frozen = regression_metrics(truth, np.concatenate(frozen_parts), groups)
        rows.append({"seed": seed, "selected": chosen, "soft_anchor": soft, "frozen_pp": frozen})
        print(f"seed={seed} soft={soft['pooled']['r2']:.4f} frozen={frozen['pooled']['r2']:.4f}", flush=True)
    def summary(key):
        v = np.array([r[key]["pooled"]["r2"] for r in rows])
        return {"pooled_r2_mean": float(v.mean()), "pooled_r2_sd": float(v.std())}
    return {"soft_anchor": summary("soft_anchor"), "frozen_pp": summary("frozen_pp"), "runs": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"))
    p.add_argument("--dataset", choices=("hust", "virkler", "nasa"), default="hust")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--anchors", default="0,0.01,0.1,1,10,100")
    p.add_argument("--decays", default="0,0.1,0.3,1")
    p.add_argument("--output", default="results/soft_anchor_ablation")
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.legacy_root).resolve()))
    from run_affine_tail_external_three import prepare_hust, prepare_virkler
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    torch.set_num_threads(2)
    seeds = tuple(map(int, args.seeds.split(',')))
    anchors = tuple(map(float, args.anchors.split(',')))
    decays = tuple(map(float, args.decays.split(',')))
    if args.dataset == "nasa":
        folds, _ = prepare_folds()
        result = evaluate_folds(folds, seeds, anchors, decays)
    else:
        prepared = prepare_hust() if args.dataset == "hust" else prepare_virkler()
        result = evaluate(*prepared[:3], seeds, anchors, decays)
    payload = {"dataset": args.dataset, "selection": "validation_mse_only",
               "result": result}
    out = Path(args.output) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
