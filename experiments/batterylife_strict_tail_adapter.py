#!/usr/bin/env python3
"""Run official BatteryLife backbones on PP's frozen battery tail splits.

The official BatteryLife task uses early complete charge/discharge curves to
predict end-of-life.  PP instead supplies a causal eight-step history at every
prediction point.  This adapter deliberately preserves the published
CPGRU/CPTransformer backbones but replaces only their data loader and final
target with PP's fixed causal RUL rows; no future curves or lifetime labels are
included in the input.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "ca-css-ncmapss"
BATTERYLIFE = ROOT / "external" / "BatteryLife"
sys.path[:0] = [str(ROOT / "src"), str(LEGACY), str(BATTERYLIFE)]

from pae_boundary_realdata import prepare_dataset  # noqa: E402
from models import CPGRU  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
MODELS = {"cpgru": CPGRU.Model}
GRID = (
    {"d_model": 16, "d_ff": 16, "e_layers": 1, "d_layers": 1, "lr": 1e-3, "wd": 0.0},
    {"d_model": 32, "d_ff": 32, "e_layers": 1, "d_layers": 1, "lr": 5e-4, "wd": 0.0},
    {"d_model": 32, "d_ff": 64, "e_layers": 2, "d_layers": 2, "lr": 5e-4, "wd": 0.1},
)


def regression_metrics(y, prediction, groups):
    """Python-3.9 compatible metric subset for the legacy Torch runtime."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    sse = float(np.sum((prediction - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    pooled = {"r2": float(1 - sse / sst) if sst else float("nan"),
              "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
              "mae": float(np.mean(np.abs(prediction - y))), "n": int(len(y))}
    per_unit = {}
    for group in np.unique(groups):
        mask = groups == group
        gy, gp = y[mask], prediction[mask]
        gsse, gsst = np.sum((gp - gy) ** 2), np.sum((gy - gy.mean()) ** 2)
        per_unit[str(group)] = {"r2": float(1 - gsse / gsst) if gsst else float("nan"), "n": int(mask.sum())}
    return {"pooled": pooled, "per_unit": per_unit,
            "unit_macro_r2": float(np.nanmean([v["r2"] for v in per_unit.values()]))}


def features(part: dict, train_center: np.ndarray | None = None, train_scale: np.ndarray | None = None):
    """Causal health, degradation-rate and relative-window-time channels."""
    window = np.asarray(part["x"], dtype=np.float32)
    relative_time = np.linspace(0.0, 1.0, window.shape[1], dtype=np.float32)[None, :, None]
    value = np.concatenate((window, np.repeat(relative_time, len(window), axis=0)), axis=2)
    if train_center is None:
        train_center = value.mean((0, 1), keepdims=True)
        train_scale = np.maximum(value.std((0, 1), keepdims=True), 1e-6)
    return ((value - train_center) / train_scale).astype(np.float32), train_center, train_scale


def prepare(name: str):
    if name == "matrb2":
        return prepare_matrb2()
    if name == "matrb2_final":
        return prepare_matrb2(final_pp=True)
    if name == "hust":
        return prepare_hust()
    raw, _ = prepare_dataset(name)
    endpoint = raw["train"]["x"][:, -1, 0]
    cutoff = float(np.quantile(endpoint, 0.25))
    masks = {"train": endpoint > cutoff, "validation": raw["val"]["x"][:, -1, 0] < cutoff}
    parts = {}
    for split, raw_split in (("train", "train"), ("validation", "val")):
        parts[split] = {key: value[masks[split]] for key, value in raw[raw_split].items()}
    parts["test"] = raw["source"]
    x_train, center, scale = features(parts["train"])
    packed = {}
    for split, part in parts.items():
        x, _, _ = features(part, center, scale)
        packed[split] = {"x": x, "y": np.asarray(part["y"], dtype=np.float32),
                         "groups": np.asarray(part["units"]).astype(str)}
    return packed


def prepare_hust():
    """HUST protocol-tail split with only prior capacity/rate observations."""
    from run_affine_tail_external_three import causal_hust_frame, evenly_spaced_per_group
    frame, _ = causal_hust_frame()
    train_frame = evenly_spaced_per_group(frame[(frame.protocol <= 6) & (frame.capacity_ah >= 1.05)].copy(), 250)
    boundary = float(train_frame.capacity_ah.min())
    selected = {
        "train": train_frame,
        "validation": evenly_spaced_per_group(frame[frame.protocol.isin([7, 8]) & (frame.capacity_ah < boundary)].copy(), 160),
        "test": evenly_spaced_per_group(frame[frame.protocol.isin([9, 10]) & (frame.capacity_ah < boundary)].copy(), 160),
    }
    by_unit = {unit: group.sort_values("step").reset_index(drop=True) for unit, group in frame.groupby("unit")}

    def pack(subset):
        history, target, groups = [], [], []
        for row in subset.itertuples(index=False):
            group = by_unit[row.unit]
            positions = np.flatnonzero(group["step"].to_numpy() == int(row.step))
            if len(positions) != 1:
                raise RuntimeError("HUST row could not be matched to its causal history")
            end = int(positions[0])
            window = group.iloc[max(0, end - 7):end + 1][["capacity_ah", "recent_rate"]].to_numpy(np.float32)
            if len(window) < 8:
                window = np.vstack((np.repeat(window[:1], 8 - len(window), axis=0), window))
            relative = np.linspace(0., 1., 8, dtype=np.float32)[:, None]
            history.append(np.hstack((window, relative)))
            target.append(float(row.RUL)); groups.append(str(row.unit))
        return {"x": np.asarray(history, dtype=np.float32), "y": np.asarray(target, dtype=np.float32),
                "groups": np.asarray(groups)}

    parts = {key: pack(value) for key, value in selected.items()}
    center = parts["train"]["x"].mean((0, 1), keepdims=True)
    scale = np.maximum(parts["train"]["x"].std((0, 1), keepdims=True), 1e-6)
    for part in parts.values():
        part["x"] = ((part["x"] - center) / scale).astype(np.float32)
    return parts


def prepare_matrb2(final_pp=False):
    """Exact 30/9/9 MATRb2 capacity-tail cohort with causal windows."""
    import h5py
    archive = ROOT / "data" / "matr" / "2017-06-30_batchdata_updated_struct_errorcorrect.mat"
    cells = []
    with h5py.File(archive, "r") as handle:
        refs = handle["batch/summary"]
        if refs.shape[0] != 48:
            raise RuntimeError("MATRb2 strict protocol requires 48 cells")
        for index in range(48):
            summary = handle[refs[index, 0]]
            cells.append((np.asarray(summary["QDischarge"]).reshape(-1).astype(np.float32),
                          np.asarray(summary["cycle"]).reshape(-1).astype(np.float32)))
    cutoff = float(np.quantile(np.concatenate([cells[index][0][7:] for index in range(30)]), .25))

    def rows(indices, boundary, is_train):
        xs, ys, groups = [], [], []
        for index in indices:
            capacity, cycle = cells[index]
            rate = np.r_[0.0, np.maximum(capacity[:-1] - capacity[1:], 0.0)]
            for end in range(7, len(capacity)):
                keep = capacity[end] > boundary if is_train else capacity[end] < boundary
                if not keep:
                    continue
                rel = np.linspace(0.0, 1.0, 8, dtype=np.float32)
                xs.append(np.column_stack((capacity[end - 7:end + 1], rate[end - 7:end + 1], rel)))
                ys.append(float(cycle[-1] - cycle[end])); groups.append("b2c%d" % index)
        return {"x": np.asarray(xs, dtype=np.float32), "y": np.asarray(ys, dtype=np.float32),
                "groups": np.asarray(groups)}

    train = rows(range(30), cutoff, True)
    boundary = float(train["x"][:, -1, 0].min())
    parts = {"train": train, "validation": rows(range(30, 39), boundary, False),
             "test": rows(range(39, 48), boundary, False)}
    if final_pp:
        # The final PP development loader reapplies the strict > boundary
        # predicate and therefore excludes the single minimum-boundary row.
        drop = int(np.argmin(parts["train"]["x"][:, -1, 0]))
        keep = np.arange(len(parts["train"]["y"])) != drop
        parts["train"] = {key: value[keep] for key, value in parts["train"].items()}
    center = parts["train"]["x"].mean((0, 1), keepdims=True)
    scale = np.maximum(parts["train"]["x"].std((0, 1), keepdims=True), 1e-6)
    for part in parts.values():
        part["x"] = ((part["x"] - center) / scale).astype(np.float32)
    return parts


def config(choice: dict):
    return SimpleNamespace(
        charge_discharge_length=1, early_cycle_threshold=8, output_num=1,
        dropout=0.0, n_heads=4, factor=1, activation="gelu", lstm_layers=1,
        **choice,
    )


def make_input(x: np.ndarray) -> torch.Tensor:
    # Official BatteryLife CP models expect [batch, cycles, variables, curve_points].
    return torch.tensor(x.transpose(0, 1, 2)[:, :, :, None], dtype=torch.float32)


def fit(model_kind: str, choice: dict, seed: int, parts: dict):
    torch.manual_seed(seed)
    if model_kind == "cptransformer":
        # BatteryLife's attention utility imports reformer-pytorch even though
        # CPTransformer itself uses full attention.  Delay that optional import
        # so the official CPGRU arm remains independently runnable.
        try:
            import reformer_pytorch  # noqa: F401
        except ImportError:
            # SelfAttention_Family imports this optional class globally although
            # CPTransformer uses FullAttention only.  Keep the official
            # CPTransformer path runnable without changing its architecture.
            optional = types.ModuleType("reformer_pytorch")
            optional.LSHSelfAttention = None
            sys.modules["reformer_pytorch"] = optional
        from models import CPTransformer
        constructor = CPTransformer.Model
    else:
        constructor = MODELS[model_kind]
    model = constructor(config(choice))
    train_x, val_x, test_x = (make_input(parts[s]["x"]) for s in ("train", "validation", "test"))
    train_mask = torch.ones((len(train_x), 8), dtype=torch.float32)
    val_mask = torch.ones((len(val_x), 8), dtype=torch.float32)
    test_mask = torch.ones((len(test_x), 8), dtype=torch.float32)
    cap = max(float(parts["train"]["y"].max()), 1.0)
    y = torch.tensor(parts["train"]["y"] / cap, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=choice["lr"], weight_decay=choice["wd"])
    rng = np.random.default_rng(seed)

    def predict(x, mask):
        with torch.no_grad():
            out = [model(batch, mask[start:start + len(batch)]).reshape(-1)
                   for start, batch in zip(range(0, len(x), 256), x.split(256))]
        return torch.cat(out).numpy() * cap

    best = float("inf")
    best_epoch = 0
    state = copy.deepcopy(model.state_dict())
    for epoch in range(1, 151):
        model.train()
        for start in range(0, len(train_x), 256):
            indices = rng.permutation(len(train_x))[start:start + 256]
            loss = torch.mean((model(train_x[indices], train_mask[indices]).reshape(-1) - y[indices]) ** 2)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        model.eval()
        validation = np.clip(predict(val_x, val_mask), 0.0, cap)
        score = float(np.mean((validation - parts["validation"]["y"]) ** 2))
        if score < best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 25:
            break
    model.load_state_dict(state); model.eval()
    return np.clip(predict(test_x, test_mask), 0.0, cap), {"validation_mse": best, "epoch": best_epoch}


def run_dataset(name: str, model_kind: str, out: Path):
    parts = prepare(name)
    choices = []
    for candidate in GRID:
        _, info = fit(model_kind, candidate, 42, parts)
        choices.append({"config": candidate, **info})
    selected = min(choices, key=lambda row: row["validation_mse"])["config"]
    predictions, rows = [], []
    for seed in SEEDS:
        prediction, info = fit(model_kind, selected, seed, parts)
        predictions.append(prediction)
        rows.append({"seed": seed, **info, "metrics": regression_metrics(parts["test"]["y"], prediction, parts["test"]["groups"])})
        print(name, model_kind, seed, rows[-1]["metrics"]["pooled"]["r2"], flush=True)
    matrix = np.asarray(predictions)
    return {"selected_config": selected, "validation_candidates": choices, "runs": rows,
            "ensemble": regression_metrics(parts["test"]["y"], matrix.mean(0), parts["test"]["groups"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="sunwoda,rwth")
    parser.add_argument("--models", default="cpgru,cptransformer")
    parser.add_argument("--output", default="results/batterylife_strict_tail_v1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    results = {"protocol": "official BatteryLife backbone; PP frozen strict split; causal 8-step health/rate/time input; validation-only selection", "datasets": {}}
    for dataset in args.datasets.split(","):
        results["datasets"][dataset] = {}
        for model in args.models.split(","):
            results["datasets"][dataset][model] = run_dataset(dataset, model, out)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
