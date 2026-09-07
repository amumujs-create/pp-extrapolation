#!/usr/bin/env python3
"""Open-loop unseen-cell MATR benchmark: direct attention vs monotone hazard PP.

Only a causal prefix ending at a fixed cycle landmark is visible.  The PP arm
predicts a discrete survival curve, so RUL is derived from a monotone learned
future process rather than regressed independently at each late-life row.
Retrospective development on the inspected MATR 2019 cohort.
"""
from __future__ import annotations

import copy
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".benchmark_deps"))
sys.path.insert(0, str(ROOT / "experiments"))

from matr_2019_latent_confirmatory import MAT
from pp_extrapolation import regression_metrics

OUT = ROOT / "results" / "matr_open_loop_conditioned_pp_v1"
LANDMARKS = (100, 150, 200, 250, 300, 350, 400)
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = (
    {"width": 24, "layers": 1, "lr": 1e-3, "wd": 1e-3},
    {"width": 32, "layers": 1, "lr": 5e-4, "wd": 1e-2},
    {"width": 48, "layers": 2, "lr": 5e-4, "wd": 1e-2},
)
HAZARD_CONFIGS = tuple(
    {"width": 32, "layers": 1, "lr": 5e-4, "wd": 1e-2, "bin_width": step, "nll_weight": weight}
    for step in (10.0, 20.0, 25.0, 40.0)
    for weight in (0.005, 0.02)
)


def matlab_text(handle, ref):
    values = np.asarray(handle[ref]).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value))


def policy_vector(text):
    return np.asarray([float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)], dtype=np.float32)


def load_cells():
    cells = []
    with h5py.File(MAT, "r") as handle:
        for index in range(handle["batch/summary"].shape[0]):
            try:
                summary = handle[handle["batch/summary"][index, 0]]
                q = np.asarray(summary["QDischarge"]).reshape(-1).astype(np.float32)
                cycle = np.asarray(summary["cycle"]).reshape(-1).astype(np.float32)
                temperature = np.asarray(summary["Tavg"]).reshape(-1).astype(np.float32)
                if not (len(q) == len(cycle) == len(temperature)) or len(q) < 24:
                    continue
                policy = matlab_text(handle, handle["batch/policy_readable"][index, 0])
                cells.append({"index": index, "q": q, "cycle": cycle, "temperature": temperature,
                              "policy": policy, "policy_vector": policy_vector(policy)})
            except Exception:
                continue
    return cells


def split_cells(cells):
    by_policy = defaultdict(list)
    for cell in cells:
        by_policy[cell["policy"]].append(cell)
    split = {"train": [], "validation": [], "test": []}
    for policy in sorted(by_policy):
        group = sorted(by_policy[policy], key=lambda cell: cell["index"])
        if len(group) < 3:
            continue
        split["train"].extend(group[:-2])
        split["validation"].append(group[-2])
        split["test"].append(group[-1])
    return split


def samples(cells, landmarks=LANDMARKS, window=64, multiscale=False):
    sequence, context, target, groups, marks = [], [], [], [], []
    for cell in cells:
        q, cycle, temp = cell["q"], cell["cycle"], cell["temperature"]
        q0 = float(np.median(q[:8]))
        soh = q / q0
        loss = np.r_[0.0, np.maximum(soh[:-1] - soh[1:], 0.0)].astype(np.float32)
        for landmark in landmarks:
            eligible = np.flatnonzero(cycle <= landmark)
            if len(eligible) < 8:
                continue
            end = int(eligible[-1])
            if end >= len(cycle) - 1:
                continue
            start = max(0, end - window + 1)
            seq = np.stack([soh[start:end + 1], loss[start:end + 1], temp[start:end + 1]], axis=-1)
            if len(seq) < window:
                seq = np.concatenate([np.repeat(seq[:1], window - len(seq), axis=0), seq], axis=0)
            sequence.append(seq)
            context_values = [*cell["policy_vector"], np.log1p(cycle[end])]
            if multiscale:
                # Every statistic ends at the prediction time.  No future value,
                # lifetime-derived fraction or test-wide normalization is used.
                for horizon in (8, 16, 32, 64, 128, None):
                    begin = 0 if horizon is None else max(0, end - horizon + 1)
                    q_slice, loss_slice, temp_slice = soh[begin:end + 1], loss[begin:end + 1], temp[begin:end + 1]
                    context_values.extend([
                        float(q_slice.mean()), float(q_slice[-1] - q_slice[0]),
                        float(loss_slice.mean()), float(loss_slice[-1] - loss_slice[0]),
                        float(temp_slice.mean()), float(temp_slice.std()),
                    ])
            context.append(context_values)
            target.append(float(cycle[-1] - cycle[end]))
            groups.append(f'c{cell["index"]}')
            marks.append(int(landmark))
    return {"sequence": np.asarray(sequence, np.float32), "context": np.asarray(context, np.float32),
            "y": np.asarray(target, np.float32), "groups": np.asarray(groups), "landmark": np.asarray(marks)}


class PrefixEncoder(nn.Module):
    def __init__(self, width, layers, context_dim):
        super().__init__()
        self.input = nn.Linear(3, width)
        self.position = nn.Parameter(torch.randn(1, 64, width) * 0.01)
        block = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.0, batch_first=True)
        self.temporal = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.context = nn.Sequential(nn.Linear(context_dim, width), nn.Tanh())

    def forward(self, sequence, context):
        states = self.temporal(self.input(sequence) + self.position)
        return states[:, -1] + self.context(context)


class DirectAttention(nn.Module):
    def __init__(self, config, context_dim):
        super().__init__()
        self.encoder = PrefixEncoder(config["width"], config["layers"], context_dim)
        self.head = nn.Linear(config["width"], 1)

    def forward(self, sequence, context):
        return self.head(self.encoder(sequence, context)).squeeze(-1)


class HazardPP(nn.Module):
    def __init__(self, config, context_dim, bins):
        super().__init__()
        self.encoder = PrefixEncoder(config["width"], config["layers"], context_dim)
        self.head = nn.Linear(config["width"], bins)

    def forward(self, sequence, context):
        hazard = torch.sigmoid(self.head(self.encoder(sequence, context)))
        survival = torch.cumprod(torch.cat([torch.ones_like(hazard[:, :1]), 1.0 - hazard[:, :-1]], 1), 1)
        return hazard, survival


class HybridHazardPP(nn.Module):
    """Monotone survival estimate with a bounded attention residual."""

    def __init__(self, config, context_dim, bins):
        super().__init__()
        self.encoder = PrefixEncoder(config["width"], config["layers"], context_dim)
        self.hazard_head = nn.Linear(config["width"], bins)
        self.residual_head = nn.Linear(config["width"], 1)
        nn.init.zeros_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias)

    def forward(self, sequence, context):
        representation = self.encoder(sequence, context)
        hazard = torch.sigmoid(self.hazard_head(representation))
        survival = torch.cumprod(torch.cat([torch.ones_like(hazard[:, :1]), 1.0 - hazard[:, :-1]], 1), 1)
        residual = 0.25 * torch.tanh(self.residual_head(representation).squeeze(-1))
        return hazard, survival, residual


def normalize(parts):
    train = parts[0]
    seq_center = train["sequence"].mean((0, 1)); seq_scale = np.maximum(train["sequence"].std((0, 1)), 1e-6)
    ctx_center = train["context"].mean(0); ctx_scale = np.maximum(train["context"].std(0), 1e-6)
    result = []
    for part in parts:
        result.append({**part, "sequence": (part["sequence"] - seq_center) / seq_scale,
                       "context": (part["context"] - ctx_center) / ctx_scale})
    return result


def fit(kind, config, seed, parts):
    train, validation, test = parts
    step = float(config.get("bin_width", 25.0))
    cap = float(np.ceil(train["y"].max() / step) * step)
    bins = int(cap / step)
    torch.manual_seed(seed)
    if kind == "direct":
        model = DirectAttention(config, train["context"].shape[1])
    elif kind == "hybrid_pp":
        model = HybridHazardPP(config, train["context"].shape[1], bins)
    else:
        model = HazardPP(config, train["context"].shape[1], bins)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    tensors = [(torch.tensor(p["sequence"]), torch.tensor(p["context"]), torch.tensor(p["y"])) for p in parts]
    unit_counts = {group: count for group, count in zip(*np.unique(train["groups"], return_counts=True))}
    weights = torch.tensor([1.0 / unit_counts[group] for group in train["groups"]], dtype=torch.float32)
    weights /= weights.mean()
    domain_codes = None
    if config.get("group_dro") and "domains" in train:
        _, codes = np.unique(train["domains"], return_inverse=True)
        domain_codes = torch.tensor(codes, dtype=torch.long)
    rng = np.random.default_rng(seed)

    def predict(which):
        model.eval(); sequence, context, _ = tensors[which]
        with torch.no_grad():
            if kind == "direct":
                value = model(sequence, context) * cap
            elif kind == "hybrid_pp":
                _, survival, residual = model(sequence, context); value = survival.sum(1) * step + residual * cap
            else:
                _, survival = model(sequence, context); value = survival.sum(1) * step
        return np.clip(value.numpy(), 0, cap)

    def validation_mse():
        prediction = predict(1)
        return float(np.mean((prediction - validation["y"]) ** 2))

    best, best_epoch, state = validation_mse(), 0, copy.deepcopy(model.state_dict())
    max_epochs = int(config.get("max_epochs", 300)); patience = int(config.get("patience", 50))
    for epoch in range(1, max_epochs + 1):
        model.train(); order = rng.permutation(len(train["y"]))
        for start in range(0, len(order), 64):
            index = torch.tensor(order[start:start + 64]); sequence, context, y = tensors[0]
            if kind == "direct":
                prediction = model(sequence[index], context[index])
                loss = torch.mean(weights[index] * (prediction - y[index] / cap) ** 2)
            else:
                if kind == "hybrid_pp":
                    hazard, survival, residual = model(sequence[index], context[index])
                    prediction = survival.sum(1) * step + residual * cap
                else:
                    hazard, survival = model(sequence[index], context[index])
                    prediction = survival.sum(1) * step
                event_bin = torch.clamp((y[index] / step).long(), max=bins - 1)
                before = torch.arange(bins)[None, :] < event_bin[:, None]
                at = torch.arange(bins)[None, :] == event_bin[:, None]
                nll = -(torch.log1p(-hazard + 1e-6) * before + torch.log(hazard + 1e-6) * at).sum(1)
                regression = ((prediction - y[index]) / cap) ** 2
                residual_penalty = 0.0 if kind != "hybrid_pp" else 0.001 * residual.square()
                sample_loss = weights[index] * (regression + float(config.get("nll_weight", 0.02)) * nll + residual_penalty)
                if domain_codes is None:
                    loss = torch.mean(sample_loss)
                else:
                    batch_domains = domain_codes[index]
                    domain_losses = torch.stack([sample_loss[batch_domains == domain].mean()
                                                 for domain in torch.unique(batch_domains)])
                    temperature = float(config.get("dro_temperature", 5.0))
                    loss = torch.logsumexp(temperature * domain_losses, 0) / temperature
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        score = validation_mse()
        if score < best - 1e-8:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break
    model.load_state_dict(state)
    return predict(2), {"validation_mse": best, "selected_epoch": best_epoch, "epochs": epoch,
                        "parameters": sum(parameter.numel() for parameter in model.parameters())}


def evaluate(kind, parts):
    search = []
    grid = HAZARD_CONFIGS if kind == "hazard_pp_tuned" else CONFIGS
    fit_kind = "hazard_pp" if kind == "hazard_pp_tuned" else kind
    for index, config in enumerate(grid):
        _, info = fit(fit_kind, config, 42, parts); search.append({"index": index, "config": config, **info})
        print("SEARCH", kind, index, info["validation_mse"], flush=True)
    selected = min(search, key=lambda row: row["validation_mse"])
    predictions, runs = [], []
    for seed in SEEDS:
        prediction, info = fit(fit_kind, selected["config"], seed, parts); predictions.append(prediction)
        metrics = regression_metrics(parts[2]["y"], prediction, parts[2]["groups"])
        runs.append({"seed": seed, **info, "metrics": metrics}); print("SEED", kind, seed, metrics["pooled"]["r2"], flush=True)
    ensemble = np.mean(predictions, axis=0)
    by_landmark = {}
    for landmark in LANDMARKS:
        mask = parts[2]["landmark"] == landmark
        if mask.sum() >= 2:
            by_landmark[str(landmark)] = regression_metrics(parts[2]["y"][mask], ensemble[mask], parts[2]["groups"][mask])
    return {"model": kind, "selected": selected, "search": search, "runs": runs,
            "ensemble": regression_metrics(parts[2]["y"], ensemble, parts[2]["groups"]), "by_landmark": by_landmark}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="direct,hazard_pp")
    parser.add_argument("--variant", choices=("standard", "multiscale"), default="standard")
    args = parser.parse_args()
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells(); split = split_cells(cells)
    multiscale = args.variant == "multiscale"
    train_landmarks = tuple(range(50, 426, 25)) if multiscale else LANDMARKS
    parts = normalize([
        samples(split["train"], landmarks=train_landmarks, multiscale=multiscale),
        samples(split["validation"], multiscale=multiscale),
        samples(split["test"], multiscale=multiscale),
    ])
    audit = {"status": "post-hoc open-loop development; no untouched claim",
             "landmarks": LANDMARKS, "train_landmarks": train_landmarks,
             "prefix_window": 64, "variant": args.variant,
             "split_indices": {name: [cell["index"] for cell in group] for name, group in split.items()},
             "counts": {name: len(part["y"]) for name, part in zip(("train", "validation", "test"), parts)}}
    destination = OUT / ("results.json" if args.variant == "standard" else "results_multiscale.json")
    existing = json.loads(destination.read_text()) if destination.exists() else {"results": {}}
    results = existing.get("results", {})
    for kind in args.models.split(","):
        results[kind] = evaluate(kind, parts)
    payload = {"audit": audit, "results": results}
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE", {kind: result["ensemble"]["pooled"]["r2"] for kind, result in results.items()}, flush=True)


if __name__ == "__main__":
    main()
