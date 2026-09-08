#!/usr/bin/env python3
"""Replay the selected final MATRb2 PP over seeds 42--46 and retain every prediction."""
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
CONFIG = {"width": 32, "learning_rate": 1e-3, "weight_decay": .1,
          "group_dro_eta": 0., "residual_decay": .05}
OUT = ROOT / "results" / "matr_batch2_pp_five_seed_replay"


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    train, validation, test = batch2()
    affine = select_affine_initialization(train, validation)
    raw, transported, runs = [], [], []
    for seed in SEEDS:
        fitted = fit_pp(train, validation, seed=seed, affine_selection=affine,
                        max_epochs=400, patience=80, **CONFIG)
        validation_prediction = predict(fitted, validation["x"])
        test_prediction = predict(fitted, test["x"])
        candidates = [{"kind": kind, "alpha": alpha,
                       "mse": float(loo(kind, alpha, validation["y"], validation_prediction,
                                        validation["x"], validation["groups"], fitted.target_scale))}
                      for kind in ("affine", "state", "rate", "all") for alpha in ALPHAS]
        selected = min(candidates, key=lambda row: row["mse"])
        a, center, scale = des(selected["kind"], validation_prediction, validation["x"])
        b, _, _ = des(selected["kind"], test_prediction, test["x"], center, scale)
        corrected = np.clip(Ridge(alpha=selected["alpha"]).fit(a, validation["y"]).predict(b),
                            0., fitted.target_scale)
        raw.append(test_prediction); transported.append(corrected)
        row = {"seed": seed, "transport": selected,
               "raw": regression_metrics(test["y"], test_prediction, test["groups"]),
               "transported": regression_metrics(test["y"], corrected, test["groups"])}
        runs.append(row)
        print(seed, row["raw"]["pooled"]["r2"], row["transported"]["pooled"]["r2"], flush=True)
    result = {"status": "frozen five-seed replay", "config": CONFIG, "seeds": list(SEEDS),
              "runs": runs,
              "raw_ensemble": regression_metrics(test["y"], np.mean(raw, axis=0), test["groups"]),
              "transported_ensemble": regression_metrics(test["y"], np.mean(transported, axis=0), test["groups"])}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", truth=test["y"], groups=test["groups"],
                        raw=np.asarray(raw), transported=np.asarray(transported))


if __name__ == "__main__":
    main()
