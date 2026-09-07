#!/usr/bin/env python3
"""Full-causal-history models on the archived MATR2019 strict capacity-tail split."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

from matr_2019_latent_confirmatory import load_cells as load_base_cells
from matr_open_loop_conditioned_pp import CONFIGS, evaluate, load_cells as load_context_cells, normalize

OUT = ROOT / "results" / "matr_strict_history_pp_v1"


def row_features(cell, end, stable=False):
    q = cell["q"].astype(np.float64); temp = cell["temperature"].astype(np.float64)
    q0 = float(np.median(q[:8])); soh = q / q0
    loss = np.r_[0.0, np.maximum(soh[:-1] - soh[1:], 0.0)]
    start = max(0, end - 63)
    if stable:
        position = np.linspace(-1.0, 0.0, end - start + 1)
        recent = np.stack([soh[start:end + 1], loss[start:end + 1], position], -1)
    else:
        recent = np.stack([soh[start:end + 1], loss[start:end + 1], temp[start:end + 1]], -1)
    if len(recent) < 64:
        recent = np.concatenate([np.repeat(recent[:1], 64 - len(recent), axis=0), recent], 0)
    context = [q0, q[end]] if stable else [*cell["policy_vector"], np.log1p(cell["cycle"][end]), q0, q[end]]
    for horizon in (8, 16, 32, 64, 128, None):
        begin = 0 if horizon is None else max(0, end - horizon + 1)
        qs, ls, ts = soh[begin:end + 1], loss[begin:end + 1], temp[begin:end + 1]
        values = [qs.mean(), qs[-1] - qs[0], ls.mean(), ls[-1] - ls[0]]
        if not stable:
            values.extend([ts.mean(), ts.std()])
        context.extend(values)
    return recent.astype(np.float32), np.asarray(context, np.float32)


def make_rows(cells, boundary, train, max_per_unit=None, stable=False):
    sequences, contexts, targets, groups, cycles, domains = [], [], [], [], [], []
    for cell in cells:
        candidates = [end for end in range(7, len(cell["q"])) if (cell["q"][end] > boundary) == train]
        if max_per_unit and len(candidates) > max_per_unit:
            positions = np.linspace(0, len(candidates) - 1, max_per_unit).round().astype(int)
            candidates = [candidates[position] for position in positions]
        for end in candidates:
            sequence, context = row_features(cell, end, stable=stable)
            sequences.append(sequence); contexts.append(context)
            targets.append(float(cell["cycle"][-1] - cell["cycle"][end])); groups.append(f'c{cell["index"]}')
            cycles.append(int(cell["cycle"][end]))
            domains.append(cell["policy"])
    return {"sequence": np.asarray(sequences, np.float32), "context": np.asarray(contexts, np.float32),
            "y": np.asarray(targets, np.float32), "groups": np.asarray(groups), "landmark": np.asarray(cycles),
            "domains": np.asarray(domains)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--group-dro", action="store_true")
    parser.add_argument("--models", default="direct,hazard_pp")
    args = parser.parse_args()
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    archived = json.loads((ROOT / "results/matr_2019_latent_confirmatory/results.json").read_text())["pretest"]
    context_cells = {cell["index"]: cell for cell in load_context_cells()}
    eligible = {cell["index"] for cell in load_base_cells()[0]}
    if not set(sum(archived["split_indices"].values(), [])).issubset(eligible & context_cells.keys()):
        raise RuntimeError("context loader changed archived eligible rows")
    cell_parts = [[context_cells[index] for index in archived["split_indices"][name]]
                  for name in ("train", "validation", "test")]
    boundary = float(archived["actual_boundary"])
    parts = normalize([make_rows(cell_parts[0], boundary, True, max_per_unit=400, stable=args.stable),
                       make_rows(cell_parts[1], boundary, False, stable=args.stable),
                       make_rows(cell_parts[2], boundary, False, stable=args.stable)])
    # Keep the same compact, predeclared architecture grid for both arms.
    for config in CONFIGS:
        config.setdefault("max_epochs", 150); config.setdefault("patience", 30)
        if args.group_dro:
            config["group_dro"] = True; config["dro_temperature"] = 5.0
    results = {kind: evaluate(kind, parts) for kind in args.models.split(",")}
    payload = {"status": "post-hoc improvement on inspected MATR2019 strict tail",
               "split_source": "results/matr_2019_latent_confirmatory/results.json",
               "boundary": boundary, "train_cap_per_unit": 400, "stable_condition_invariant": args.stable,
               "group_dro": args.group_dro,
               "counts": {name: len(part["y"]) for name, part in zip(("train", "validation", "test"), parts)},
               "results": results}
    suffix = "_stable" if args.stable else ""
    suffix += "_group_dro" if args.group_dro else ""
    destination = OUT / ("results" + suffix + ".json")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE", {kind: result["ensemble"]["pooled"]["r2"] for kind, result in results.items()}, flush=True)


if __name__ == "__main__":
    main()
