#!/usr/bin/env python3
"""Support-adaptive PP on the frozen FD002+FD004 strict OP-hull split."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

from pp_extrapolation.model import fit_pp, predict, select_affine_initialization


def pp_split(data):
    return {"x": data["x"], "y": data["y"], "groups": data["units"].astype(str)}


def append_exact_hull_distance(train, val, tests, legacy_modules):
    """Append a train-only OP-hull distance; train rows are exactly zero."""
    (FDS, load_fd, split_units, filter_table, OperatingConditionNormalizer,
     make_condition_rows, audit_convex_hull_support) = legacy_modules
    val_distances, test_distances = [], {}
    for fd in FDS:
        table, _, _ = load_fd(fd)
        train_units, validation_units, test_units = split_units(table, seed=42)
        train_unit_table = table[np.isin(table[:, 0].astype(np.int32), train_units)]
        q70, q85 = np.quantile(train_unit_table[:, 2], [0.70, 0.85])
        selected_train = filter_table(table, train_units, lambda value: value <= q70)
        selected_validation = filter_table(table, validation_units, lambda value: value > q85)
        selected_test = filter_table(table, test_units, lambda value: value > q85)
        normalizer = OperatingConditionNormalizer.fit(selected_train, 5, seed=42)
        validation_rows = make_condition_rows(selected_validation, table, normalizer, stride=1)
        test_rows = make_condition_rows(selected_test, table, normalizer, stride=1)
        val_distances.append(audit_convex_hull_support(
            selected_train[:, 2:5], validation_rows["op"], feature_names=("OP1","OP2","OP3")
        ).distance)
        test_distances[fd] = audit_convex_hull_support(
            selected_train[:, 2:5], test_rows["op"], feature_names=("OP1","OP2","OP3")
        ).distance
    train["x"] = np.column_stack([train["x"], np.zeros(len(train["y"]))])
    val["x"] = np.column_stack([val["x"], np.concatenate(val_distances)])
    for fd in FDS:
        tests[fd]["x"] = np.column_stack([tests[fd]["x"], test_distances[fd]])
    return {"validation_min": float(np.min(np.concatenate(val_distances))),
            "validation_median": float(np.median(np.concatenate(val_distances))),
            "test_median": {fd: float(np.median(test_distances[fd])) for fd in FDS}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"))
    p.add_argument("--output", default="results/cmapss_support_adaptive")
    args = p.parse_args()
    legacy = Path(args.legacy_root).resolve(); sys.path.insert(0, str(legacy))
    from run_pae_machine_strict_hull import FDS, aggregate
    from run_pae_machine_strict_hull import filter_table, make_condition_rows, split_units
    from run_pae_machine_strict_hull_v4 import prepare_hull_validation
    from cmapss_prior_off import load_fd
    from cmapss_fleet_nn import OperatingConditionNormalizer
    from pae_extrapolation_audit import audit_convex_hull_support
    torch.set_num_threads(2)
    train0, val0, tests0, audit, transform = prepare_hull_validation()
    distance_audit = append_exact_hull_distance(
        train0, val0, tests0,
        (FDS, load_fd, split_units, filter_table, OperatingConditionNormalizer,
         make_condition_rows, audit_convex_hull_support),
    )
    train, val = pp_split(train0), pp_split(val0)
    affine = select_affine_initialization(train, val)
    anchors, decays = (0.0, 0.1, 100.0), (0.0, 0.1, 0.3, 1.0)
    candidates = []
    for anchor in anchors:
        for decay in decays:
            fit = fit_pp(train, val, seed=42, affine_selection=affine,
                         affine_anchor_weight=anchor, residual_decay=decay)
            candidates.append({"anchor": anchor, "decay": decay,
                               "validation_mse": fit.selection["best_validation_mse"]})
            print(f"select anchor={anchor:g} decay={decay:g} val={candidates[-1]['validation_mse']:.3f}", flush=True)
    selected = min(candidates, key=lambda r: (r["validation_mse"], -r["anchor"], r["decay"]))
    runs = []
    for seed in (42,43,44,45,46):
        fit = fit_pp(train, val, seed=seed, affine_selection=affine,
                     affine_anchor_weight=selected["anchor"], residual_decay=selected["decay"])
        preds = {fd: predict(fit, tests0[fd]["x"]) for fd in FDS}
        metrics = aggregate(tests0, preds)
        runs.append({"seed": seed, "metrics": metrics, "selection": fit.selection})
        print(f"seed={seed} pooled={metrics['pooled']['r2']:.4f}", flush=True)
    values = np.array([r["metrics"]["pooled"]["r2"] for r in runs])
    payload = {"experiment":"cmapss_support_adaptive", "selection_seed":42,
               "selected":selected, "candidates":candidates, "split_audit":audit,
               "transform":transform, "distance_audit":distance_audit,
               "pooled_r2_mean":float(values.mean()),
               "pooled_r2_sd":float(values.std()), "runs":runs}
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(payload,indent=2)+"\n")

if __name__ == "__main__": main()
