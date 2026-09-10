#!/usr/bin/env python3
"""Local TabPFN v3 on the frozen MICH PP battery split.

Uses the same adapter as the common competitor suite (prepare_battery) and the
same local TabPFN contract as Sunwoda/RWTH/MATRb2: CPU v3, seeds 42-46, one
estimator, max 1,000 equal-unit training rows, no test-label selection.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "ca-css-ncmapss"
OUT = ROOT / "results" / "final_tabpfn_mich_extension_v1"
SEEDS = (42, 43, 44, 45, 46)
MAX_TRAIN = 1_000

fake_mlx = types.ModuleType("mlx")
fake_mlx_core = types.ModuleType("mlx.core")
fake_mlx.core = fake_mlx_core
sys.modules.setdefault("mlx", fake_mlx)
sys.modules.setdefault("mlx.core", fake_mlx_core)
os.environ.setdefault("TABPFN_DEVICE", "cpu")
os.environ.setdefault("TORCH_DEVICE", "cpu")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

sys.path[:0] = [str(LEGACY)]


def regression_metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    def one(target: np.ndarray, estimate: np.ndarray) -> dict:
        sse = float(np.sum((estimate - target) ** 2))
        sst = float(np.sum((target - target.mean()) ** 2))
        return {
            "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean((estimate - target) ** 2))),
            "mae": float(np.mean(np.abs(estimate - target))),
            "n": int(len(target)),
        }

    per_unit = {str(unit): one(y[groups == unit], prediction[groups == unit]) for unit in np.unique(groups)}
    return {
        "pooled": one(y, prediction),
        "unit_macro_r2": float(np.nanmean([value["r2"] for value in per_unit.values()])),
        "per_unit": per_unit,
    }


def equal_unit_subsample(data: dict, maximum: int = MAX_TRAIN) -> tuple[dict, dict]:
    n = len(data["y"])
    if n <= maximum:
        return data, {"n_original": n, "n_used": n, "method": "all rows"}
    groups = data["groups"].astype(str)
    units = np.unique(groups)
    quota, remainder = divmod(maximum, len(units))
    chosen: list[int] = []
    for position, unit in enumerate(units):
        local = np.flatnonzero(groups == unit)
        count = min(len(local), quota + int(position < remainder))
        take = np.unique(np.linspace(0, len(local) - 1, count).round().astype(int))
        chosen.extend(local[take].tolist())
    idx = np.asarray(sorted(chosen[:maximum]), dtype=np.int64)
    return ({key: value[idx] for key, value in data.items()}, {
        "n_original": n,
        "n_used": int(len(idx)),
        "n_units": int(len(units)),
        "method": "equal-unit deterministic evenly-spaced rows",
    })


def battery_features(part: dict, cycle_scale: float) -> dict:
    z = np.asarray(part["x"], dtype=np.float32)
    health, rate = z[:, :, 0], z[:, :, 1]
    x = np.column_stack((
        health[:, -1], rate[:, -1], health.mean(1), rate.mean(1),
        health[:, -1] - health[:, 0], rate[:, -1] - rate[:, 0],
        part["cycles"] / cycle_scale,
    )).astype(np.float32)
    return {"x": x, "y": np.asarray(part["y"], dtype=np.float32),
            "groups": np.asarray(part["units"]).astype(str)}


def prepare_mich() -> tuple[dict, dict, dict]:
    from pae_boundary_realdata import prepare_dataset

    raw, _ = prepare_dataset("mich")
    endpoint = raw["train"]["x"][:, -1, 0]
    cutoff = float(np.quantile(endpoint, 0.25))
    train_mask = endpoint > cutoff
    val_mask = raw["val"]["x"][:, -1, 0] < cutoff
    cycle_scale = max(float(raw["train"]["cycles"].max()), 1.0)
    train = battery_features({key: value[train_mask] for key, value in raw["train"].items()}, cycle_scale)
    val = battery_features({key: value[val_mask] for key, value in raw["val"].items()}, cycle_scale)
    test = battery_features(raw["source"], cycle_scale)
    return train, val, test


def predict(train: dict, x_test: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion("v3"), device="cpu", random_state=seed,
        ignore_pretraining_limits=True,
    )
    model.fit(train["x"], train["y"])
    chunks = [model.predict(x_test[start:start + 256]) for start in range(0, len(x_test), 256)]
    cap = max(float(np.max(train["y"])), 1.0)
    return np.clip(np.concatenate(chunks), 0.0, cap), "local:cpu:v3"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing results: {result_path}")

    train, validation, test = prepare_mich()
    if len(test["y"]) != 202:
        raise RuntimeError(f"locked MICH test must have 202 rows, found {len(test['y'])}")
    used_train, sampling = equal_unit_subsample(train)
    predictions, rows = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        prediction, backend = predict(used_train, test["x"], seed)
        metrics = regression_metrics(test["y"], prediction, test["groups"])
        predictions.append(prediction)
        rows.append({
            "seed": seed,
            "backend": backend,
            "runtime_seconds": time.perf_counter() - started,
            "metrics": metrics,
        })
        print("mich", seed, f"pooled_r2={metrics['pooled']['r2']:.6f}", flush=True)

    matrix = np.asarray(predictions)
    ensemble = regression_metrics(test["y"], matrix.mean(0), test["groups"])
    single = np.asarray([row["metrics"]["pooled"]["r2"] for row in rows])
    np.savez_compressed(OUT / "predictions_mich.npz", truth=test["y"], groups=test["groups"], predictions=matrix)
    payload = {
        "experiment": "final_tabpfn_mich_extension_v1",
        "model": "local TabPFN v3 CPU",
        "protocol": "same prepare_battery split as V-REx/GroupDRO/Engression; TabPFN contract matches tabpfn_external_batteries_v1",
        "seeds": list(SEEDS),
        "max_train": MAX_TRAIN,
        "selection": "no test-label model selection; fixed TabPFN v3 configuration",
        "mich": {
            "n": {"train": len(train["y"]), "validation": len(validation["y"]), "test": len(test["y"])},
            "sampling": sampling,
            "single_seed_pooled_r2_mean": float(single.mean()),
            "single_seed_pooled_r2_sd": float(single.std()),
            "ensemble_metrics": ensemble,
            "runs": rows,
        },
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("ensemble", ensemble["pooled"]["r2"], flush=True)


if __name__ == "__main__":
    main()
