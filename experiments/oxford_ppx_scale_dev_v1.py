#!/usr/bin/env python3
"""Development: validation-only scale executor on the locked Oxford split.

Oxford already had test labels opened.  This is not a confirmatory cohort.
It tests whether a lifetime-scale / val-only scale route repairs Cell8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from oxford_untouched import load_capacity, sha256, MAT
from oxford_outcome_heldout_v2 import feats
from pp_extrapolation import (
    fit_latent_regime_pp,
    fit_lifetime_scale_pp,
    predict_latent_regime,
    predict_lifetime_scale,
    regression_metrics,
    select_affine_initialization,
)

OUT = ROOT / "results" / "oxford_ppx_scale_dev_v1"
SEEDS = (42, 43, 44, 45, 46)
PRESET = "short"
SEP = 0.01
CUT = 0.8396755456924438


def rows(cells, ids, cut, side):
    xs, ys, groups, elapsed = [], [], [], []
    for i in ids:
        h = np.asarray(cells[i] / cells[i][0], float)
        x, ix = feats(h, PRESET)
        mask = h[ix] > cut if side == "train" else h[ix] < cut
        xs.append(x[mask])
        ys.append((len(h) - 1 - ix[mask]).astype(np.float32))
        elapsed.append(ix[mask].astype(np.float32) + 1.0)
        groups += [f"Cell{i}"] * int(mask.sum())
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "elapsed": np.concatenate(elapsed),
        "groups": np.asarray(groups),
    }


def val_scale(pred, y):
    denom = float(np.dot(pred, pred))
    if denom <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(pred, y) / denom, 0.25, 4.0))


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError(f"Refusing to overwrite {OUT / 'results.json'}")

    cells = load_capacity()
    train = rows(cells, range(1, 5), CUT, "train")
    val = rows(cells, (5, 6), CUT, "tail")
    test = rows(cells, (7, 8), CUT, "tail")
    if abs(CUT - 0.8396755456924438) > 1e-9 or len(val["y"]) < 8 or len(test["y"]) != 56:
        raise RuntimeError("locked Oxford split mismatch")
    affine = select_affine_initialization(train, val)
    cap = max(float(train["y"].max()), 1.0)

    routes = {"latent": [], "lifetime": [], "latent_val_scale": []}
    search = []
    for seed in SEEDS:
        latent = fit_latent_regime_pp(
            train, val, seed=seed, affine_selection=affine,
            max_epochs=400, patience=80, separation_weight=SEP, gate_weight=0.0, width=32,
        )
        p_val_lat = predict_latent_regime(latent, val["x"])
        p_te_lat = np.clip(predict_latent_regime(latent, test["x"]), 0, cap)
        life = fit_lifetime_scale_pp(
            train, val, train["elapsed"], val["elapsed"], seed=seed,
            width=32, learning_rate=1e-3, weight_decay=0.1,
        )
        p_val_life = predict_lifetime_scale(life, val["x"], val["elapsed"])
        p_te_life = np.clip(predict_lifetime_scale(life, test["x"], test["elapsed"]), 0, cap)
        scale = val_scale(p_val_lat, val["y"])
        p_val_scaled = p_val_lat * scale
        p_te_scaled = np.clip(p_te_lat * scale, 0, cap)
        row = {
            "seed": seed,
            "latent_val_mse": float(np.mean((p_val_lat - val["y"]) ** 2)),
            "lifetime_val_mse": float(np.mean((p_val_life - val["y"]) ** 2)),
            "scaled_val_mse": float(np.mean((p_val_scaled - val["y"]) ** 2)),
            "val_scale": scale,
        }
        search.append(row)
        routes["latent"].append(p_te_lat)
        routes["lifetime"].append(p_te_life)
        routes["latent_val_scale"].append(p_te_scaled)
        print("SEED", seed, {k: round(row[k], 4) for k in row if k != "seed"}, flush=True)

    mean_val = {
        "latent": float(np.mean([r["latent_val_mse"] for r in search])),
        "lifetime": float(np.mean([r["lifetime_val_mse"] for r in search])),
        "latent_val_scale": float(np.mean([r["scaled_val_mse"] for r in search])),
    }
    chosen = min(mean_val, key=mean_val.get)
    y, g = test["y"], test["groups"]
    summary = {}
    for name, preds in routes.items():
        mat = np.asarray(preds)
        summary[name] = {
            "validation_mse_mean": mean_val[name],
            "ensemble": regression_metrics(y, mat.mean(0), g),
            "per_seed": [regression_metrics(y, p, g) for p in mat],
        }
    np.savez_compressed(
        OUT / "predictions.npz", y=y, groups=g,
        latent=np.asarray(routes["latent"]),
        lifetime=np.asarray(routes["lifetime"]),
        latent_val_scale=np.asarray(routes["latent_val_scale"]),
    )
    payload = {
        "status": "post-test development on already-scored Oxford split; not confirmatory",
        "mechanism": "Cell8 lifetime-scale mismatch; validation-only lifetime PP and global scale",
        "archive_sha256": sha256(MAT),
        "n": {"train": len(train["y"]), "validation": len(val["y"]), "test": len(y)},
        "baseline_locked_latent_r2": -0.11934027429040661,
        "search": search,
        "validation_route_mse": mean_val,
        "selected_by_validation": chosen,
        "routes": summary,
        "success_vs_locked_baseline": bool(summary[chosen]["ensemble"]["pooled"]["r2"] > 0),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("CHOSEN", chosen, "R2", summary[chosen]["ensemble"]["pooled"]["r2"], flush=True)
    for name in routes:
        ens = summary[name]["ensemble"]
        print(name, "pooled", round(ens["pooled"]["r2"], 3),
              "C7", round(ens["per_unit"]["Cell7"]["r2"], 3),
              "C8", round(ens["per_unit"]["Cell8"]["r2"], 3), flush=True)


if __name__ == "__main__":
    main()
