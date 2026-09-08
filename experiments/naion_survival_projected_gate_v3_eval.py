#!/usr/bin/env python3
"""Prospective survival-support PP and prefix gate on Na-ion group 3."""
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

DATA = ROOT / "data" / "naion_external_v3"
OUT = ROOT / "results" / "naion_survival_projected_gate_v3"
SEEDS = (42, 43, 44, 45, 46)


def load_series():
    result = {}
    for path in sorted(DATA.glob("270040-3-*.csv")):
        frame = pd.read_csv(path, encoding="gbk")
        cycle_col, cap_col = _find(frame.columns, "循环号"), _find(frame.columns, "放电容量")
        clean = pd.DataFrame({
            "cycle": pd.to_numeric(frame[cycle_col], errors="coerce"),
            "capacity": pd.to_numeric(frame[cap_col].replace("-", np.nan), errors="coerce"),
        }).dropna()
        clean = clean[(clean.cycle >= 0) & (clean.capacity > 0)]
        grouped = clean.groupby("cycle", sort=True).capacity.max()
        t, c = grouped.index.to_numpy(float), grouped.to_numpy(float)
        if len(t) >= 40:
            result[path.name] = {"cycle": t, "capacity": c}
    return result


def row_ages(series, ids, boundary, side):
    ages = []
    for uid in ids:
        t = series[uid]["cycle"]
        span = t - t[0]
        mask = span <= boundary if side == "prefix" else span > boundary
        ages.extend(span[np.flatnonzero(mask & (np.arange(len(t)) >= 3))].tolist())
    return np.asarray(ages, float)


def project(predictions, ages, known_max_life):
    return np.clip(np.asarray(predictions), 0.0, np.maximum(0.0, known_max_life - ages))


def main():
    torch.set_num_threads(2)
    series = load_series()
    all_ids = sorted(series)
    if len(all_ids) != 8:
        raise RuntimeError(f"locked cohort requires exactly 8 eligible files, found {all_ids}")
    ids = {"train": all_ids[:4], "validation": all_ids[4:6], "test": all_ids[6:]}
    life = {u: float(series[u]["cycle"][-1] - series[u]["cycle"][0]) for u in all_ids}
    train_life = [life[u] for u in ids["train"]]
    boundary = float(np.floor(0.60 * np.median(train_life)))
    time_scale = float(max(train_life))
    tr = make_rows(series, ids["train"], boundary, time_scale, "prefix")
    va = make_rows(series, ids["validation"], boundary, time_scale, "tail")
    va_age = row_ages(series, ids["validation"], boundary, "tail")
    affine = select_affine_initialization(tr, va)
    plain_models, pp_models, pval_raw, qval_raw = [], [], [], []
    for seed in SEEDS:
        fm = fit_plain(tr, va, seed=seed)
        fp = fit_pp(tr, va, seed=seed, affine_selection=affine)
        plain_models.append(fm); pp_models.append(fp)
        pval_raw.append(predict_plain(fm, va["x"])); qval_raw.append(predict(fp, va["x"]))
    pval_raw, qval_raw = np.asarray(pval_raw), np.asarray(qval_raw)
    pval = project(pval_raw, va_age, max(train_life))
    qval = project(qval_raw, va_age, max(train_life))
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
    gate = {
        "protocol": "protocols/NAION_SURVIVAL_PROJECTED_GATE_V3_PROTOCOL.md",
        "split_ids": ids, "boundary_train_cycle_span": boundary,
        "known_max_life_validation": max(train_life),
        "validation_projected": {"plain": pvm, "pp": qvm,
            "relative_mse_gain": relative_gain, "normalized_seed_disagreement": disagreement},
        "validation_raw": {
            "plain": regression_metrics(va["y"], pval_raw.mean(0), va["groups"]),
            "pp": regression_metrics(va["y"], qval_raw.mean(0), va["groups"]),
        },
        "test_prefix_distance": dict(zip(ids["test"], distances.tolist())),
        "checks": checks, "approved": bool(all(checks.values())),
        "test_prefix_commitment_sha256": hashlib.sha256(json.dumps({
            "test_ids": ids["test"], "test_prefix_descriptors": test_desc.tolist()
        }, sort_keys=True).encode()).hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_decision_preoutcome.json").write_text(json.dumps(gate, indent=2) + "\n")

    te = make_rows(series, ids["test"], boundary, time_scale, "tail")
    te_age = row_ages(series, ids["test"], boundary, "tail")
    known_test_life = max(life[u] for u in ids["train"] + ids["validation"])
    praw = np.asarray([predict_plain(m, te["x"]) for m in plain_models])
    qraw = np.asarray([predict(m, te["x"]) for m in pp_models])
    ptest, qtest = project(praw, te_age, known_test_life), project(qraw, te_age, known_test_life)
    pm = regression_metrics(te["y"], ptest.mean(0), te["groups"])
    qm = regression_metrics(te["y"], qtest.mean(0), te["groups"])
    success = bool(qm["pooled"]["r2"] > 0 and qm["pooled"]["r2"] >= pm["pooled"]["r2"])
    result = {
        "status": "prospective one-shot external confirmation",
        "development_source": "group-2 tail explosion; no group-3 outcome used",
        "gate": gate, "known_max_life_test": known_test_life,
        "n": {"train": len(tr["y"]), "validation": len(va["y"]), "test": len(te["y"])},
        "raw": {"plain": regression_metrics(te["y"], praw.mean(0), te["groups"]),
                "pp": regression_metrics(te["y"], qraw.mean(0), te["groups"])},
        "projected": {"plain": pm, "pp": qm},
        "pp_gain_r2": qm["pooled"]["r2"] - pm["pooled"]["r2"],
        "actual_pp_success": success, "gate_correct": bool(gate["approved"] == success),
        "selective_outcome": "PP" if gate["approved"] else "ABSTAIN",
        "seed_runs_projected": [{"seed": seed,
            "plain_r2": regression_metrics(te["y"], ptest[i], te["groups"])["pooled"]["r2"],
            "pp_r2": regression_metrics(te["y"], qtest[i], te["groups"])["pooled"]["r2"]}
            for i, seed in enumerate(SEEDS)],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=te["y"], groups=te["groups"],
                        plain_raw=praw, pp_raw=qraw, plain=ptest, pp=qtest, age=te_age)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
