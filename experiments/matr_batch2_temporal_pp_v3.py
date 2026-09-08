#!/usr/bin/env python3
"""Final-PP temporal residual with perturbation-consistency training."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]

from batterylife_strict_tail_adapter import prepare_matrb2  # noqa: E402
from group_robust_pp import batch2  # noqa: E402
from matr_batch2_regime_transport_pp import ALPHAS, des, loo  # noqa: E402
from pp_extrapolation import fit_pp, predict, regression_metrics, select_affine_initialization  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
BASE = {"width": 32, "learning_rate": 1e-3, "weight_decay": 0.1,
        "group_dro_eta": 0.0, "residual_decay": 0.05}
TEMPORAL = tuple({"width": 64, "bound": 0.5, "decay": 0.05, "lr": 5e-4, "wd": 0.1,
                  "noise": noise, "consistency": consistency}
                 for noise, consistency in ((0.0, 0.0), (0.01, 0.1), (0.01, 1.0),
                                             (0.03, 0.1), (0.03, 1.0), (0.03, 5.0),
                                             (0.05, 0.1), (0.05, 1.0), (0.05, 5.0)))


def weights(groups):
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    value = 1.0 / count[inverse]
    return torch.tensor(value / value.mean(), dtype=torch.float32)


class TemporalResidual(nn.Module):
    def __init__(self, config, support_min):
        super().__init__()
        self.gru = nn.GRU(3, config["width"], batch_first=True)
        self.head = nn.Sequential(nn.Linear(config["width"], config["width"]), nn.Tanh(),
                                  nn.Linear(config["width"], 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.bound = float(config["bound"]); self.decay = float(config["decay"])
        self.register_buffer("support_min", torch.tensor(float(support_min), dtype=torch.float32))

    def forward(self, sequence, base):
        state = self.gru(sequence)[0][:, -1]
        residual = self.bound * torch.tanh(self.head(state).squeeze(1) / self.bound)
        if self.decay:
            # Capacity channel is standardized on training only; positive distance
            # below the observed tail boundary marks outward extrapolation.
            distance = torch.relu(self.support_min - sequence[:, -1, 0])
            residual = residual * torch.exp(-self.decay * distance)
        return base + residual


def fit_temporal(config, seed, sequence, base, target, group, validation):
    torch.manual_seed(seed); model = TemporalResidual(config, np.min(sequence[:, -1, 0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    x = torch.tensor(sequence, dtype=torch.float32)
    b = torch.tensor(base, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    w = weights(group); rng = np.random.default_rng(seed)
    vx = torch.tensor(validation[0], dtype=torch.float32); vb = torch.tensor(validation[1], dtype=torch.float32)

    def val_prediction():
        model.eval()
        with torch.no_grad(): return torch.cat([model(vx[i:i + 512], vb[i:i + 512]) for i in range(0, len(vx), 512)]).numpy()

    best, best_epoch, state = float("inf"), 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, 251):
        model.train(); order = rng.permutation(len(x))
        for start in range(0, len(x), 512):
            index = torch.tensor(order[start:start + 512], dtype=torch.long)
            clean = model(x[index], b[index])
            loss = torch.mean(w[index] * (clean - y[index]) ** 2)
            if config["consistency"]:
                noise_a = torch.randn_like(x[index]) * config["noise"]
                noise_b = torch.randn_like(x[index]) * config["noise"]
                paired = (model(x[index] + noise_a, b[index]) - model(x[index] + noise_b, b[index])) ** 2
                loss = loss + config["consistency"] * torch.mean(w[index] * paired)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        prediction = val_prediction(); score = float(np.mean((prediction - validation[2]) ** 2))
        if score < best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 40: break
    model.load_state_dict(state); model.eval()
    return model, {"validation_mse": best, "epoch": best_epoch}


def infer(model, sequence, base, cap):
    x = torch.tensor(sequence, dtype=torch.float32); b = torch.tensor(base, dtype=torch.float32)
    with torch.no_grad(): value = torch.cat([model(x[i:i + 512], b[i:i + 512]) for i in range(0, len(x), 512)]).numpy()
    return np.clip(value * cap, 0.0, cap)


def fit_base(parts, affine, seed):
    return fit_pp(parts[0], parts[1], seed=seed, affine_selection=affine, max_epochs=400, patience=80, **BASE)


def main():
    torch.set_num_threads(2)
    output = ROOT / "results" / "matr_batch2_temporal_pp_v3"; output.mkdir(parents=True, exist_ok=True)
    train, validation, test = batch2(); parts = (train, validation, test)
    sequence = prepare_matrb2(final_pp=True)
    for split, row in (("train", train), ("validation", validation), ("test", test)):
        if not np.array_equal(sequence[split]["groups"].astype(str), row["groups"].astype(str)) or not np.allclose(sequence[split]["y"], row["y"]):
            raise RuntimeError("sequence/PP row alignment failed for " + split)
    affine = select_affine_initialization(train, validation)
    selection_base = fit_base(parts, affine, 42); cap = float(selection_base.target_scale)
    base_train = predict(selection_base, train["x"]) / cap
    base_validation = predict(selection_base, validation["x"]) / cap
    search = []
    for config in TEMPORAL:
        _, info = fit_temporal(config, 42, sequence["train"]["x"], base_train, train["y"] / cap,
                               train["groups"], (sequence["validation"]["x"], base_validation, validation["y"] / cap))
        search.append({"config": config, **info}); print("SEARCH", config, info["validation_mse"], flush=True)
    chosen = min(search, key=lambda row: row["validation_mse"])["config"]
    raw_matrix, transported_matrix, runs = [], [], []
    for seed in SEEDS:
        base_fit = selection_base if seed == 42 else fit_base(parts, affine, seed)
        bt, bv, be = (predict(base_fit, split["x"]) / cap for split in parts)
        model, info = fit_temporal(chosen, seed, sequence["train"]["x"], bt, train["y"] / cap,
                                   train["groups"], (sequence["validation"]["x"], bv, validation["y"] / cap))
        vp = infer(model, sequence["validation"]["x"], bv, cap)
        tp = infer(model, sequence["test"]["x"], be, cap)
        candidates = [{"kind": kind, "alpha": alpha, "mse": float(loo(kind, alpha, validation["y"], vp,
                                                                         validation["x"], validation["groups"], cap))}
                      for kind in ("affine", "state", "rate", "all") for alpha in ALPHAS]
        selected = min(candidates, key=lambda row: row["mse"])
        a, mean, scale = des(selected["kind"], vp, validation["x"])
        b, _, _ = des(selected["kind"], tp, test["x"], mean, scale)
        transported = np.clip(Ridge(alpha=selected["alpha"]).fit(a, validation["y"]).predict(b), 0.0, cap)
        raw_matrix.append(tp); transported_matrix.append(transported)
        runs.append({"seed": seed, **info, "transport": selected,
                     "raw": regression_metrics(test["y"], tp, test["groups"]),
                     "transported": regression_metrics(test["y"], transported, test["groups"])})
        print("REFIT", seed, runs[-1]["raw"]["pooled"]["r2"], runs[-1]["transported"]["pooled"]["r2"], flush=True)
    result = {"status": "post-hoc development", "model": "final PP + consistency-trained bounded causal GRU residual",
              "base_config": BASE, "temporal_search": search, "selected_temporal": chosen, "runs": runs,
              "raw_ensemble": regression_metrics(test["y"], np.mean(raw_matrix, 0), test["groups"]),
              "transported_ensemble": regression_metrics(test["y"], np.mean(transported_matrix, 0), test["groups"])}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", truth=test["y"], groups=test["groups"],
                        raw=np.asarray(raw_matrix), transported=np.asarray(transported_matrix))
    print("FINAL", result["raw_ensemble"]["pooled"]["r2"], result["transported_ensemble"]["pooled"]["r2"])


if __name__ == "__main__": main()
