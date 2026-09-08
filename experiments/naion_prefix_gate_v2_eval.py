#!/usr/bin/env python3
"""Prospective v2 relative-skill gate on untouched Na-ion group 2."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from naion_prefix_gate_eval import _find, compatibility, make_rows, prefix_descriptor
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, regression_metrics, select_affine_initialization

DATA = ROOT / "data" / "naion_external_v2"
OUT = ROOT / "results" / "naion_prefix_gate_v2"
SEEDS = (42, 43, 44, 45, 46)


def load_series():
    series = {}
    for path in sorted(DATA.glob("270040-2-*-12.csv")):
        frame = pd.read_csv(path, encoding="gbk")
        cycle_col = _find(frame.columns, "循环号")
        cap_col = _find(frame.columns, "放电容量")
        clean = pd.DataFrame({
            "cycle": pd.to_numeric(frame[cycle_col], errors="coerce"),
            "capacity": pd.to_numeric(frame[cap_col].replace("-", np.nan), errors="coerce"),
        }).dropna()
        clean = clean[(clean.cycle >= 0) & (clean.capacity > 0)]
        grouped = clean.groupby("cycle", sort=True).capacity.max()
        t, c = grouped.index.to_numpy(float), grouped.to_numpy(float)
        if len(t) >= 40:
            series[path.name] = {"cycle": t, "capacity": c}
    return series


def main():
    torch.set_num_threads(2)
    series = load_series()
    ids_all = sorted(series)
    if len(ids_all) != 8:
        raise RuntimeError(f"locked cohort requires exactly 8 eligible files, found {ids_all}")
    ids = {"train": ids_all[:4], "validation": ids_all[4:6], "test": ids_all[6:]}
    train_spans = [series[u]["cycle"][-1] - series[u]["cycle"][0] for u in ids["train"]]
    boundary = float(np.floor(0.60 * np.median(train_spans)))
    time_scale = float(max(train_spans))
    tr = make_rows(series, ids["train"], boundary, time_scale, "prefix")
    va = make_rows(series, ids["validation"], boundary, time_scale, "tail")
    affine = select_affine_initialization(tr, va)

    plain_models, pp_models, pval, qval = [], [], [], []
    for seed in SEEDS:
        plain_model = fit_plain(tr, va, seed=seed)
        pp_model = fit_pp(tr, va, seed=seed, affine_selection=affine)
        plain_models.append(plain_model)
        pp_models.append(pp_model)
        pval.append(predict_plain(plain_model, va["x"]))
        qval.append(predict(pp_model, va["x"]))
    pval, qval = np.asarray(pval), np.asarray(qval)
    pvm = regression_metrics(va["y"], pval.mean(0), va["groups"])
    qvm = regression_metrics(va["y"], qval.mean(0), va["groups"])
    relative_gain = 1.0 - qvm["pooled"]["rmse"] ** 2 / max(pvm["pooled"]["rmse"] ** 2, 1e-12)
    disagreement = float(np.mean(np.std(qval, axis=0)) / max(np.std(va["y"]), 1e-8))
    train_desc = np.asarray([prefix_descriptor(series[u]["cycle"], series[u]["capacity"], boundary) for u in ids["train"]])
    test_desc = np.asarray([prefix_descriptor(series[u]["cycle"], series[u]["capacity"], boundary) for u in ids["test"]])
    distances = compatibility(train_desc, test_desc)
    checks = {
        "validation_mse_gain_at_least_2pct": bool(relative_gain >= 0.02),
        "seed_disagreement_at_most_0_25": bool(disagreement <= 0.25),
        "enough_units": bool(len(ids["train"]) >= 4 and len(ids["validation"]) >= 2),
        "all_prefix_distances_at_most_3": bool(np.all(distances <= 3.0)),
    }
    commitment = hashlib.sha256(json.dumps({
        "test_ids": ids["test"], "test_prefix_descriptors": test_desc.tolist()
    }, sort_keys=True).encode()).hexdigest()
    gate = {
        "protocol": "protocols/NAION_PREFIX_GATE_V2_PROTOCOL.md",
        "split_ids": ids,
        "boundary_train_cycle_span": boundary,
        "validation": {"plain": pvm, "pp": qvm, "relative_mse_gain": relative_gain,
                       "normalized_seed_disagreement": disagreement},
        "test_prefix_distance": dict(zip(ids["test"], distances.tolist())),
        "checks": checks,
        "approved": bool(all(checks.values())),
        "test_prefix_commitment_sha256": commitment,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_decision_preoutcome.json").write_text(json.dumps(gate, indent=2) + "\n")

    te = make_rows(series, ids["test"], boundary, time_scale, "tail")
    ptest = np.asarray([predict_plain(m, te["x"]) for m in plain_models])
    qtest = np.asarray([predict(m, te["x"]) for m in pp_models])
    pm = regression_metrics(te["y"], ptest.mean(0), te["groups"])
    qm = regression_metrics(te["y"], qtest.mean(0), te["groups"])
    success = bool(qm["pooled"]["r2"] > 0 and qm["pooled"]["r2"] >= pm["pooled"]["r2"])
    result = {
        "status": "prospective one-shot external confirmation",
        "development_source": "Na-ion group 1 v1 false rejection; no group-2 outcome used",
        "gate": gate,
        "n": {"train": len(tr["y"]), "validation": len(va["y"]), "test": len(te["y"])},
        "plain_ensemble": pm,
        "pp_ensemble": qm,
        "pp_gain_r2": qm["pooled"]["r2"] - pm["pooled"]["r2"],
        "actual_pp_success": success,
        "gate_correct": bool(gate["approved"] == success),
        "selective_outcome": "PP" if gate["approved"] else "ABSTAIN",
        "seed_runs": [{
            "seed": seed,
            "plain_r2": regression_metrics(te["y"], ptest[i], te["groups"])["pooled"]["r2"],
            "pp_r2": regression_metrics(te["y"], qtest[i], te["groups"])["pooled"]["r2"],
        } for i, seed in enumerate(SEEDS)],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=te["y"], groups=te["groups"], plain=ptest, pp=qtest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
