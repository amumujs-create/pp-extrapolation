#!/usr/bin/env python3
"""Post-confirmation repair of PP on the axial-fan endpoint task.

This is explicitly a development analysis: official test labels were already
observed in the preceding frozen confirmation.  It repairs the task mismatch by
training on the last 105 time steps and uses one internally gated PP network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from axial_fan_untouched_confirmation import CONFIGS, DATA, load_features, metric
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, select_affine_initialization

OUT = ROOT / "results" / "axial_fan_tail_gate_pp_development_v1"
SEEDS = (42, 43, 44, 45, 46)
TAIL_HORIZON = 105.0
PP_CONFIG = {
    "width": 32,
    "learning_rate": 5e-4,
    "weight_decay": 2.0,
    "learned_affine_gate": True,
    "direct_residual_mixture": True,
}
FROZEN_REFERENCE = {
    "1P_8F": {"pp_r2": -0.375, "mlp_r2": -0.068},
    "4P_1F": {"pp_r2": -1.780, "mlp_r2": -1.349},
    "4P_8F": {"pp_r2": -0.289, "mlp_r2": -0.132},
}


def tail(rows: dict) -> dict:
    keep = np.asarray(rows["y"]) <= TAIL_HORIZON
    return {key: np.asarray(value)[keep] for key, value in rows.items()}


def bootstrap_gain(y, pp, mlp, groups, replicates=20000):
    units = np.unique(groups)
    by_unit = {u: np.flatnonzero(groups == u) for u in units}
    rng = np.random.default_rng(20260909)
    gain = np.empty(replicates)
    for b in range(replicates):
        chosen = rng.choice(units, len(units), replace=True)
        index = np.concatenate([by_unit[u] for u in chosen])
        pp_rmse = np.mean((pp[index] - y[index]) ** 2) ** 0.5
        mlp_rmse = np.mean((mlp[index] - y[index]) ** 2) ** 0.5
        gain[b] = mlp_rmse - pp_rmse
    return {
        "mlp_rmse_minus_pp_rmse": float(np.mean(gain)),
        "95ci": [float(x) for x in np.quantile(gain, (0.025, 0.975))],
        "probability_positive": float(np.mean(gain > 0)),
        "replicates": int(replicates),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "post-confirmation model development; not untouched evidence",
        "repair": "endpoint-matched last-105-step risk set plus one gated PP network",
        "tail_horizon": TAIL_HORIZON,
        "pp_config": PP_CONFIG,
        "seeds": list(SEEDS),
        "frozen_failure_reference": FROZEN_REFERENCE,
        "results": {},
    }
    pooled = {"truth": [], "pp": [], "mlp": [], "groups": []}

    for tag, n_regimes in CONFIGS.items():
        train, validation, full, endpoint, *_ = load_features(tag, n_regimes)
        train, validation, full = tail(train), tail(validation), tail(full)
        truth = np.loadtxt(DATA / "sealed" / f"RUL_FAN_{tag}.txt").reshape(-1)
        affine_selection = select_affine_initialization(train, validation)
        pp_predictions, mlp_predictions, runs = [], [], []
        for seed in SEEDS:
            selected_pp = fit_pp(
                train, validation, seed=seed, affine_selection=affine_selection,
                max_epochs=450, patience=70, **PP_CONFIG,
            )
            pp_epoch = max(int(selected_pp.selection["selected_epoch"]), 1)
            final_affine = select_affine_initialization(
                full, full, alphas=(affine_selection["selected_alpha"],)
            )
            final_pp = fit_pp(
                full, full, seed=seed, affine_selection=final_affine,
                max_epochs=pp_epoch, patience=pp_epoch + 1, **PP_CONFIG,
            )
            selected_mlp = fit_plain(
                train, validation, seed=seed, max_epochs=450, patience=70
            )
            mlp_epoch = max(int(selected_mlp["selected_epoch"]), 1)
            final_mlp = fit_plain(
                full, full, seed=seed, max_epochs=mlp_epoch,
                patience=mlp_epoch + 1, restore_best=False,
            )
            pp_prediction = predict(final_pp, endpoint["x"])
            mlp_prediction = predict_plain(final_mlp, endpoint["x"])
            pp_predictions.append(pp_prediction)
            mlp_predictions.append(mlp_prediction)
            runs.append({
                "seed": seed,
                "pp_epoch": pp_epoch,
                "mlp_epoch": mlp_epoch,
                "pp": metric(truth, pp_prediction),
                "plain_mlp": metric(truth, mlp_prediction),
            })
            print(tag, seed, pp_epoch, mlp_epoch, flush=True)

        pp_prediction = np.mean(pp_predictions, axis=0)
        mlp_prediction = np.mean(mlp_predictions, axis=0)
        inference = bootstrap_gain(
            truth, pp_prediction, mlp_prediction, endpoint["groups"]
        )
        payload["results"][tag] = {
            "n_train_tail_rows": len(train["y"]),
            "n_validation_tail_rows": len(validation["y"]),
            "n_test_fans": len(truth),
            "pp": metric(truth, pp_prediction),
            "plain_mlp": metric(truth, mlp_prediction),
            "paired_unit_bootstrap": inference,
            "pp_rmse_win": bool(
                metric(truth, pp_prediction)["rmse"]
                < metric(truth, mlp_prediction)["rmse"]
            ),
            "runs": runs,
        }
        np.savez_compressed(
            OUT / f"{tag}_predictions.npz", truth=truth,
            pp=np.asarray(pp_predictions), mlp=np.asarray(mlp_predictions),
            groups=endpoint["groups"],
        )
        pooled["truth"].append(truth)
        pooled["pp"].append(pp_prediction)
        pooled["mlp"].append(mlp_prediction)
        pooled["groups"].append(np.asarray([f"{tag}:{g}" for g in endpoint["groups"]]))

    truth = np.concatenate(pooled["truth"])
    pp_prediction = np.concatenate(pooled["pp"])
    mlp_prediction = np.concatenate(pooled["mlp"])
    groups = np.concatenate(pooled["groups"])
    payload["aggregate"] = {
        "pp": metric(truth, pp_prediction),
        "plain_mlp": metric(truth, mlp_prediction),
        "macro_r2": {
            name: float(np.mean([
                payload["results"][tag][name]["r2"] for tag in CONFIGS
            ])) for name in ("pp", "plain_mlp")
        },
        "pp_rmse_wins": int(sum(
            payload["results"][tag]["pp_rmse_win"] for tag in CONFIGS
        )),
        "paired_unit_bootstrap": bootstrap_gain(
            truth, pp_prediction, mlp_prediction, groups
        ),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
