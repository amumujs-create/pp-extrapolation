#!/usr/bin/env python3
"""Local TabPFN v3 on frozen Sunwoda, RWTH, and MATR batch-2 splits.

CPU inference is capped at 1,000 equal-unit sampled training rows.  This
matches the existing local-TabPFN protocol and is reported as a supplementary
same-information comparison, not as a full-data TabPFN result.
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
OUT = ROOT / "results" / "tabpfn_external_batteries_v1"
SEEDS = (42, 43, 44, 45, 46)
MAX_TRAIN = 1_000

# The installed local TabPFN package imports MLX even for CPU inference.  The
# desktop headless host has no Metal device.  CPU tensors never reach this
# backend, so a minimal import shim keeps the requested CPU execution intact.
fake_mlx = types.ModuleType("mlx")
fake_mlx_core = types.ModuleType("mlx.core")
fake_mlx.core = fake_mlx_core
sys.modules.setdefault("mlx", fake_mlx)
sys.modules.setdefault("mlx.core", fake_mlx_core)
os.environ.setdefault("TABPFN_DEVICE", "cpu")
os.environ.setdefault("TORCH_DEVICE", "cpu")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

sys.path[:0] = [str(ROOT / "experiments"), str(LEGACY)]


def regression_metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict:
    """Metric subset kept local for the Python 3.9 TabPFN runtime."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - y

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
    """Keep an evenly spaced quota per training unit without using labels."""
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


def prepare_sunwoda_rwth(name: str) -> tuple[dict, dict, dict]:
    from pae_boundary_realdata import prepare_dataset

    raw, _ = prepare_dataset(name)
    endpoint = raw["train"]["x"][:, -1, 0]
    cutoff = float(np.quantile(endpoint, 0.25))
    train_mask = endpoint > cutoff
    val_mask = raw["val"]["x"][:, -1, 0] < cutoff
    cycle_scale = max(float(raw["train"]["cycles"].max()), 1.0)
    train = battery_features({key: value[train_mask] for key, value in raw["train"].items()}, cycle_scale)
    val = battery_features({key: value[val_mask] for key, value in raw["val"].items()}, cycle_scale)
    test = battery_features(raw["source"], cycle_scale)
    return train, val, test


def prepare_matr_batch2() -> tuple[dict, dict, dict]:
    # Kept here rather than importing the confirmatory runner: that runner
    # imports the Python 3.10 PP package, whereas the local TabPFN runtime is
    # Python 3.9.  The data access and frozen 30/9/9 cohort protocol are
    # identical to ``matr_batch2_confirmatory.py``.
    import h5py

    archive = ROOT / "data" / "matr" / "2017-06-30_batchdata_updated_struct_errorcorrect.mat"
    cells = []
    with h5py.File(archive, "r") as handle:
        refs = handle["batch/summary"]
        if refs.shape[0] != 48:
            raise RuntimeError(f"locked protocol requires 48 cells, found {refs.shape[0]}")
        for index in range(48):
            summary = handle[refs[index, 0]]
            cells.append((
                np.asarray(summary["QDischarge"]).reshape(-1).astype(np.float32),
                np.asarray(summary["cycle"]).reshape(-1).astype(np.float32),
            ))

    def make_rows(ids, boundary, train=False, targets=True):
        xs, ys, groups, coordinates = [], [], [], []
        for index in ids:
            capacity, cycle = cells[index]
            rate = np.r_[0.0, np.maximum(capacity[:-1] - capacity[1:], 0.0)]
            for end in range(7, len(capacity)):
                keep = capacity[end] > boundary if train else capacity[end] < boundary
                if not keep:
                    continue
                health = capacity[end - 7:end + 1]
                window_rate = rate[end - 7:end + 1]
                xs.append([health[-1], window_rate[-1], health.mean(), window_rate.mean(),
                           health[-1] - health[0], window_rate[-1] - window_rate[0]])
                coordinates.append(health[-1])
                groups.append(f"b2c{index}")
                if targets:
                    ys.append(float(cycle[-1] - cycle[end]))
        out = {"x": np.asarray(xs, dtype=np.float32), "groups": np.asarray(groups),
               "coordinate": np.asarray(coordinates, dtype=np.float32)}
        if targets:
            out["y"] = np.asarray(ys, dtype=np.float32)
        return out

    endpoint = np.concatenate([cells[index][0][7:] for index in range(30)])
    cutoff = float(np.quantile(endpoint, 0.25))
    train = make_rows(range(30), cutoff, train=True)
    boundary = float(train["coordinate"].min())
    val = make_rows(range(30, 39), boundary)
    test = make_rows(range(39, 48), boundary, targets=True)
    return (
        {"x": train["x"], "y": train["y"], "groups": train["groups"].astype(str)},
        {"x": val["x"], "y": val["y"], "groups": val["groups"].astype(str)},
        {"x": test["x"], "y": test["y"], "groups": test["groups"].astype(str)},
    )


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


def run(name: str, prepared: tuple[dict, dict, dict]) -> dict:
    train, validation, test = prepared
    used_train, sampling = equal_unit_subsample(train)
    predictions, rows = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        prediction, backend = predict(used_train, test["x"], seed)
        metrics = regression_metrics(test["y"], prediction, test["groups"])
        predictions.append(prediction)
        rows.append({"seed": seed, "backend": backend, "runtime_seconds": time.perf_counter() - started, "metrics": metrics})
        print(name, seed, f"pooled_r2={metrics['pooled']['r2']:.6f}", flush=True)
    matrix = np.asarray(predictions)
    ensemble = regression_metrics(test["y"], matrix.mean(0), test["groups"])
    single = np.asarray([row["metrics"]["pooled"]["r2"] for row in rows])
    np.savez_compressed(OUT / f"predictions_{name}.npz", truth=test["y"], groups=test["groups"], predictions=matrix)
    return {
        "n": {"train": len(train["y"]), "validation": len(validation["y"]), "test": len(test["y"])},
        "sampling": sampling,
        "single_seed_pooled_r2_mean": float(single.mean()),
        "single_seed_pooled_r2_sd": float(single.std()),
        "ensemble_metrics": ensemble,
        "runs": rows,
    }


def write_report(payload: dict) -> None:
    labels = {"sunwoda": "Sunwoda", "rwth": "RWTH", "matrb2": "MATR batch 2"}
    lines = [
        "# Local TabPFN v3: external battery extrapolation", "",
        "TabPFN v3 used CPU inference, one estimator per seed, five seeds (42–46), and a maximum of 1,000 equal-unit sampled training rows. The frozen test rows and pooled R² metric match the PP splits. This is a supplementary capped-data comparison.", "",
        "| dataset | TabPFN seed mean pooled R² | TabPFN ensemble pooled R² | train rows used | test rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("sunwoda", "rwth", "matrb2"):
        result = payload["datasets"][key]
        lines.append(
            f"| {labels[key]} | {result['single_seed_pooled_r2_mean']:.3f}±{result['single_seed_pooled_r2_sd']:.3f} | "
            f"{result['ensemble_metrics']['pooled']['r2']:.3f} | {result['sampling']['n_used']} | {result['n']['test']} |"
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing results: {result_path}")
    datasets = {
        "sunwoda": run("sunwoda", prepare_sunwoda_rwth("sunwoda")),
        "rwth": run("rwth", prepare_sunwoda_rwth("rwth")),
        "matrb2": run("matrb2", prepare_matr_batch2()),
    }
    payload = {
        "experiment": "tabpfn_external_batteries_v1",
        "model": "local TabPFN v3 CPU",
        "seeds": list(SEEDS),
        "max_train": MAX_TRAIN,
        "selection": "no test-label model selection; fixed TabPFN v3 configuration",
        "datasets": datasets,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(payload)


if __name__ == "__main__":
    main()
