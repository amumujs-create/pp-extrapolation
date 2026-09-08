#!/usr/bin/env python3
"""Temporal PP development on the fixed MATR batch-2 strict-tail split.

The affine extrapolation path remains explicit. A causal GRU sees the same
eight observed capacity/rate points as the BatteryLife CPGRU comparison and
can only add a bounded, support-decayed residual. Model/configuration and the
optional output transport are selected with validation units only.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
from batterylife_strict_tail_adapter import prepare_matrb2, regression_metrics  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CONFIGS = tuple(
    {"width": width, "lr": lr, "wd": wd, "bound": bound, "decay": decay}
    for width, lr, wd in ((16, 1e-3, 0.0), (32, 5e-4, 0.1), (64, 5e-4, 0.1))
    for bound, decay in ((0.10, 0.0), (0.25, 0.0), (0.25, 0.05), (0.50, 0.05))
)


def summaries(sequence):
    health, rate = sequence[:, :, 0], sequence[:, :, 1]
    return np.column_stack((health[:, -1], rate[:, -1], health.mean(1), rate.mean(1),
                            health[:, -1] - health[:, 0], rate[:, -1] - rate[:, 0])).astype(np.float32)


def group_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    value = 1.0 / counts[inverse]
    return (value / value.mean()).astype(np.float32)


def choose_affine(parts):
    train_x = summaries(parts["train"]["x"])
    validation_x = summaries(parts["validation"]["x"])
    weights = group_weights(parts["train"]["groups"])
    rows = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(train_x, parts["train"]["y"], sample_weight=weights)
        prediction = np.clip(model.predict(validation_x), 0.0, parts["train"]["y"].max())
        rows.append({"alpha": alpha, "mse": float(np.mean((prediction - parts["validation"]["y"]) ** 2))})
    chosen = min(rows, key=lambda row: row["mse"])["alpha"]
    model = Ridge(alpha=chosen).fit(train_x, parts["train"]["y"], sample_weight=weights)
    return model, rows


class TemporalPP(nn.Module):
    def __init__(self, width, bound, decay, support_min, support_max):
        super().__init__()
        self.gru = nn.GRU(3, width, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.bound = float(bound); self.decay = float(decay)
        self.register_buffer("support_min", torch.tensor(support_min, dtype=torch.float32))
        self.register_buffer("support_max", torch.tensor(support_max, dtype=torch.float32))

    def forward(self, sequence, affine, summary):
        encoded = self.gru(sequence)[0][:, -1]
        residual = self.bound * torch.tanh(self.head(encoded).squeeze(1) / self.bound)
        if self.decay:
            distance = torch.linalg.vector_norm(torch.relu(self.support_min - summary) +
                                                torch.relu(summary - self.support_max), dim=1)
            residual = residual * torch.exp(-self.decay * distance)
        return affine + residual


def train(parts, affine_model, config, seed):
    cap = max(float(parts["train"]["y"].max()), 1.0)
    arrays = {}
    for split, part in parts.items():
        sx = summaries(part["x"])
        arrays[split] = {
            "sequence": torch.tensor(part["x"], dtype=torch.float32),
            "summary": torch.tensor(sx, dtype=torch.float32),
            "affine": torch.tensor(np.clip(affine_model.predict(sx), 0.0, cap) / cap, dtype=torch.float32),
            "y": torch.tensor(part["y"] / cap, dtype=torch.float32),
        }
    torch.manual_seed(seed)
    train_summary = arrays["train"]["summary"]
    model = TemporalPP(config["width"], config["bound"], config["decay"],
                       train_summary.amin(0), train_summary.amax(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    weights = torch.tensor(group_weights(parts["train"]["groups"]), dtype=torch.float32)
    rng = np.random.default_rng(seed)

    def predict(split):
        item = arrays[split]
        with torch.no_grad():
            values = [model(item["sequence"][start:start + 512], item["affine"][start:start + 512],
                            item["summary"][start:start + 512])
                      for start in range(0, len(item["y"]), 512)]
        return np.clip(torch.cat(values).numpy() * cap, 0.0, cap)

    best, best_epoch, state = float("inf"), 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, 301):
        model.train(); order = rng.permutation(len(arrays["train"]["y"]))
        for start in range(0, len(order), 512):
            index = torch.tensor(order[start:start + 512], dtype=torch.long)
            item = arrays["train"]
            output = model(item["sequence"][index], item["affine"][index], item["summary"][index])
            loss = torch.mean(weights[index] * (output - item["y"][index]) ** 2)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        model.eval(); validation = predict("validation")
        score = float(np.mean((validation - parts["validation"]["y"]) ** 2))
        if score < best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 50:
            break
    model.load_state_dict(state); model.eval()
    return predict("validation"), predict("test"), {"validation_mse": best, "epoch": best_epoch}


def design(kind, prediction, summary, mean=None, scale=None):
    columns = {"prediction": (), "state": (0, 1), "rate": (2, 3), "all": tuple(range(summary.shape[1]))}[kind]
    extra = summary[:, columns] if columns else np.empty((len(summary), 0))
    if mean is None:
        mean, scale = extra.mean(0), np.maximum(extra.std(0), 1e-8)
    return np.column_stack((prediction, (extra - mean) / scale)), mean, scale


def loo_mse(kind, alpha, prediction, summary, truth, groups, cap):
    errors = []
    for unit in np.unique(groups):
        fit = groups != unit
        a, mean, scale = design(kind, prediction[fit], summary[fit])
        b, _, _ = design(kind, prediction[~fit], summary[~fit], mean, scale)
        estimate = np.clip(Ridge(alpha=alpha).fit(a, truth[fit]).predict(b), 0.0, cap)
        errors.extend((estimate - truth[~fit]) ** 2)
    return float(np.mean(errors))


def main():
    torch.set_num_threads(2)
    out = ROOT / "results" / "matr_batch2_temporal_pp_v1"; out.mkdir(parents=True, exist_ok=True)
    parts = prepare_matrb2(); affine, affine_search = choose_affine(parts)
    search = []
    for config in CONFIGS:
        validation, _, info = train(parts, affine, config, 42)
        search.append({"config": config, **info})
        print("SEARCH", config, info["validation_mse"], flush=True)
    chosen = min(search, key=lambda row: row["validation_mse"])["config"]
    validation_summary = summaries(parts["validation"]["x"])
    test_summary = summaries(parts["test"]["x"])
    raw, transported, runs = [], [], []
    for seed in SEEDS:
        validation, test, info = train(parts, affine, chosen, seed)
        candidates = [{"kind": kind, "alpha": alpha,
                       "mse": loo_mse(kind, alpha, validation, validation_summary,
                                       parts["validation"]["y"], parts["validation"]["groups"],
                                       parts["train"]["y"].max())}
                      for kind in ("prediction", "state", "rate", "all") for alpha in ALPHAS]
        selected = min(candidates, key=lambda row: row["mse"])
        a, mean, scale = design(selected["kind"], validation, validation_summary)
        b, _, _ = design(selected["kind"], test, test_summary, mean, scale)
        calibrated = np.clip(Ridge(alpha=selected["alpha"]).fit(a, parts["validation"]["y"]).predict(b),
                             0.0, parts["train"]["y"].max())
        raw.append(test); transported.append(calibrated)
        runs.append({"seed": seed, **info, "transport": selected,
                     "raw": regression_metrics(parts["test"]["y"], test, parts["test"]["groups"]),
                     "transported": regression_metrics(parts["test"]["y"], calibrated, parts["test"]["groups"])})
        print("REFIT", seed, runs[-1]["raw"]["pooled"]["r2"],
              runs[-1]["transported"]["pooled"]["r2"], flush=True)
    result = {"status": "post-hoc development", "model": "affine path + bounded support-decayed causal GRU residual",
              "selection": "validation only", "affine_search": affine_search, "search": search,
              "selected_config": chosen, "runs": runs,
              "raw_ensemble": regression_metrics(parts["test"]["y"], np.mean(raw, 0), parts["test"]["groups"]),
              "transported_ensemble": regression_metrics(parts["test"]["y"], np.mean(transported, 0), parts["test"]["groups"])}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "predictions.npz", truth=parts["test"]["y"], groups=parts["test"]["groups"],
                        raw=np.asarray(raw), transported=np.asarray(transported))
    print("FINAL", result["raw_ensemble"]["pooled"]["r2"],
          result["transported_ensemble"]["pooled"]["r2"], flush=True)


if __name__ == "__main__":
    main()
