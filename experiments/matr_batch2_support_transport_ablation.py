#!/usr/bin/env python3
"""Matched 2x2 ablation of support decay and regime transport on final MATRb2 PP."""
import json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]
from group_robust_pp import batch2
from matr_batch2_regime_transport_pp import ALPHAS, des, loo
from pp_extrapolation import fit_pp, predict, regression_metrics, select_affine_initialization

SEEDS = (42, 43, 44, 45, 46)
OUT = ROOT / "results" / "matr_batch2_support_transport_ablation_v1"


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    train, validation, test = batch2(); affine = select_affine_initialization(train, validation)
    result = {"status": "post-hoc matched 2x2 component ablation", "seeds": list(SEEDS), "arms": {}}
    arrays = {"truth": test["y"], "groups": test["groups"]}
    for decay in (0.0, 0.05):
        raw, transported, runs = [], [], []
        for seed in SEEDS:
            fitted = fit_pp(train, validation, seed=seed, affine_selection=affine, width=32,
                            learning_rate=1e-3, weight_decay=.1, group_dro_eta=0.,
                            residual_decay=decay, max_epochs=400, patience=80)
            vp, tp = predict(fitted, validation["x"]), predict(fitted, test["x"])
            candidates = [{"kind": kind, "alpha": alpha,
                           "mse": float(loo(kind, alpha, validation["y"], vp, validation["x"],
                                            validation["groups"], fitted.target_scale))}
                          for kind in ("affine", "state", "rate", "all") for alpha in ALPHAS]
            selected = min(candidates, key=lambda row: row["mse"])
            a, center, scale = des(selected["kind"], vp, validation["x"])
            b, _, _ = des(selected["kind"], tp, test["x"], center, scale)
            corrected = np.clip(Ridge(alpha=selected["alpha"]).fit(a, validation["y"]).predict(b),
                                0., fitted.target_scale)
            raw.append(tp); transported.append(corrected)
            runs.append({"seed": seed, "transport": selected,
                         "raw": regression_metrics(test["y"], tp, test["groups"]),
                         "transported": regression_metrics(test["y"], corrected, test["groups"])})
            print(decay, seed, runs[-1]["raw"]["pooled"]["r2"],
                  runs[-1]["transported"]["pooled"]["r2"], flush=True)
        key = "decay_%.2f" % decay
        result["arms"][key] = {"runs": runs,
            "raw_ensemble": regression_metrics(test["y"], np.mean(raw, 0), test["groups"]),
            "transported_ensemble": regression_metrics(test["y"], np.mean(transported, 0), test["groups"])}
        arrays[key + "_raw"] = np.asarray(raw); arrays[key + "_transported"] = np.asarray(transported)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
