#!/usr/bin/env python3
"""Policy-stratified MATR split sensitivity audit and equal-budget PP/FT benchmark.

This is retrospective method development on an already inspected cohort.  It is
intended to separate health-tail extrapolation from an accidental charging-policy
shift in the archived file-order split.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".benchmark_deps"))
sys.path.insert(0, str(ROOT / "experiments"))

from rtdl_revisiting_models import FTTransformer
from extended_nn_benchmark import GRID, train
from matr_2019_latent_confirmatory import MAT, load_cells
from pp_extrapolation import regression_metrics, select_affine_initialization, support_distance

OUT = ROOT / "results" / "matr_split_sensitivity_v1"


def _matlab_string(handle: h5py.File, ref) -> str:
    values = np.asarray(handle[ref]).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value))


def load_policy_map() -> dict[int, str]:
    result = {}
    with h5py.File(MAT, "r") as handle:
        for index in range(handle["batch/policy_readable"].shape[0]):
            result[index] = _matlab_string(handle, handle["batch/policy_readable"][index, 0])
    return result


def policy_stratified_cells():
    cells, excluded, _ = load_cells()
    policies = load_policy_map()
    by_policy = defaultdict(list)
    for cell in cells:
        by_policy[policies[cell["index"]]].append(cell)
    split = {"train": [], "validation": [], "test": []}
    for policy in sorted(by_policy):
        group = sorted(by_policy[policy], key=lambda item: item["index"])
        if len(group) < 3:
            raise RuntimeError(f"policy {policy!r} has fewer than three eligible cells")
        split["validation"].append(group[-2])
        split["test"].append(group[-1])
        split["train"].extend(group[:-2])
    return split, policies, excluded


def normalized_rows(cells, boundary: float, *, train: bool, include_age: bool):
    xs, ys, groups, coordinate, ages = [], [], [], [], []
    for cell in cells:
        q = np.asarray(cell["q"], dtype=np.float64)
        cycle = np.asarray(cell["cycle"], dtype=np.float64)
        q0 = float(np.median(q[:8]))
        soh = q / q0
        rate = np.r_[0.0, np.maximum(soh[:-1] - soh[1:], 0.0)]
        for end in range(7, len(q)):
            if (soh[end] > boundary) != train:
                continue
            history, losses = soh[end - 7 : end + 1], rate[end - 7 : end + 1]
            features = [
                history[-1],
                losses[-1],
                history.mean(),
                losses.mean(),
                history[-1] - history[0],
                losses[-1] - losses[0],
            ]
            if include_age:
                features.append(np.log1p(cycle[end]))
            xs.append(features)
            ys.append(float(cycle[-1] - cycle[end]))
            groups.append(f'c{cell["index"]}')
            coordinate.append(soh[end])
            ages.append(cycle[end])
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "groups": np.asarray(groups),
        "coordinate": np.asarray(coordinate, dtype=np.float32),
        "age": np.asarray(ages, dtype=np.float32),
    }


def make_split(include_age: bool):
    split_cells, policies, excluded = policy_stratified_cells()
    endpoints = np.concatenate(
        [np.asarray(cell["q"])[7:] / np.median(np.asarray(cell["q"])[:8]) for cell in split_cells["train"]]
    )
    nominal_boundary = float(np.quantile(endpoints, 0.25))
    initial_train = normalized_rows(split_cells["train"], nominal_boundary, train=True, include_age=include_age)
    boundary = float(initial_train["coordinate"].min())
    parts = [
        normalized_rows(split_cells[name], boundary, train=name == "train", include_age=include_age)
        for name in ("train", "validation", "test")
    ]
    train_rows, validation_rows, test_rows = parts
    _, validation_hull = support_distance(
        train_rows["coordinate"][:, None], validation_rows["coordinate"][:, None]
    )
    _, test_hull = support_distance(train_rows["coordinate"][:, None], test_rows["coordinate"][:, None])
    audit = {
        "status": "post-hoc split sensitivity analysis; not confirmatory",
        "definition": "policy-stratified unseen cells and relative-SOH future tail",
        "include_current_age": include_age,
        "excluded": excluded,
        "boundary": boundary,
        "counts": {name: len(rows["y"]) for name, rows in zip(("train", "validation", "test"), parts)},
        "split_indices": {name: [cell["index"] for cell in cells] for name, cells in split_cells.items()},
        "policy_counts": {
            name: dict(Counter(policies[cell["index"]] for cell in cells)) for name, cells in split_cells.items()
        },
        "hull": {"validation": validation_hull.summary(), "test": test_hull.summary()},
    }
    return parts, audit


class EndpointAttention(nn.Module):
    """Attention estimates latent EOL; the RUL head enforces time translation."""

    def __init__(self, dimension: int, config: dict):
        super().__init__()
        self.encoder = FTTransformer(
            n_cont_features=dimension,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=config["depth"],
            d_block=config["width"],
            attention_n_heads=4,
            attention_dropout=0.0,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=2.0,
            ffn_dropout=0.0,
            residual_dropout=0.0,
        )

    def forward(self, features, age_fraction):
        latent_eol = self.encoder(features, None).squeeze(-1)
        return latent_eol - age_fraction


def train_endpoint(config, seed, parts, affine):
    import copy

    from pp_extrapolation.model import equal_group_weights, transform_features

    train_rows, validation_rows, test_rows = parts
    arrays = [transform_features(rows["x"], affine["center"], affine["scale"]) for rows in parts]
    features = [torch.tensor(values, dtype=torch.float32) for values in arrays]
    target_scale = float(affine["target_scale"])
    ages = [torch.tensor(rows["age"] / target_scale, dtype=torch.float32) for rows in parts]
    target = torch.tensor(train_rows["y"] / target_scale, dtype=torch.float32)
    weights = torch.tensor(equal_group_weights(train_rows["groups"]), dtype=torch.float32)
    torch.manual_seed(seed)
    model = EndpointAttention(features[0].shape[1], config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    rng = np.random.default_rng(seed)

    def validation_mse():
        model.eval()
        with torch.no_grad():
            chunks = []
            for x_chunk, age_chunk in zip(features[1].split(512), ages[1].split(512)):
                chunks.append(model(x_chunk, age_chunk))
            prediction = np.clip(torch.cat(chunks).numpy() * target_scale, 0, target_scale)
        return float(np.mean((prediction - validation_rows["y"]) ** 2))

    best, best_epoch, state = validation_mse(), 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, 151):
        model.train()
        order = rng.permutation(len(features[0]))
        for start in range(0, len(order), 512):
            index = torch.tensor(order[start : start + 512])
            prediction = model(features[0][index], ages[0][index])
            loss = torch.mean(weights[index] * (prediction - target[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        score = validation_mse()
        if score < best - 1e-10:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 25:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        raw = torch.cat(
            [model(x_chunk, age_chunk) for x_chunk, age_chunk in zip(features[2].split(512), ages[2].split(512))]
        ).numpy() * target_scale
    return np.clip(raw, 0, target_scale), {
        "validation_mse": best,
        "selected_epoch": best_epoch,
        "epochs": epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run(view: str, kind: str):
    include_age = view == "health_age"
    parts, audit = make_split(include_age)
    train_rows, validation_rows, test_rows = parts
    affine = select_affine_initialization(train_rows, validation_rows)
    search = []
    for index, config in enumerate(GRID):
        if kind == "endpoint_attention":
            _, info = train_endpoint(config, 42, parts, affine)
        else:
            _, _, info = train(kind, config, 42, parts, affine)
        search.append({"index": index, "config": config, **info})
        print("SEARCH", view, kind, index, info["validation_mse"], flush=True)
    chosen = min(search, key=lambda row: row["validation_mse"])
    predictions, runs = [], []
    for seed in range(42, 47):
        if kind == "endpoint_attention":
            prediction, info = train_endpoint(chosen["config"], seed, parts, affine)
        else:
            model, test_tensor, info = train(kind, chosen["config"], seed, parts, affine)
            with torch.no_grad():
                raw = torch.cat([model(batch) for batch in test_tensor.split(512)]).numpy() * affine["target_scale"]
            prediction = np.clip(raw, 0, affine["target_scale"])
        predictions.append(prediction)
        runs.append({"seed": seed, **info, "metrics": regression_metrics(test_rows["y"], prediction, test_rows["groups"])})
        print("REFIT", view, kind, seed, runs[-1]["metrics"]["pooled"]["r2"], flush=True)
    result = {
        "audit": audit,
        "model": kind,
        "selected": chosen,
        "search": search,
        "runs": runs,
        "ensemble": regression_metrics(test_rows["y"], np.mean(predictions, axis=0), test_rows["groups"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{view}_{kind}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("DONE", destination, result["ensemble"]["pooled"]["r2"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("health", "health_age"), required=True)
    parser.add_argument("--model", choices=("pp_joint", "ft_transformer", "endpoint_attention"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.view, args.model)


if __name__ == "__main__":
    main()
