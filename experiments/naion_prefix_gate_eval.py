#!/usr/bin/env python3
"""Prospective Na-ion evaluation with a pre-outcome prefix compatibility gate."""
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

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import (
    fit_pp,
    predict,
    regression_metrics,
    select_affine_initialization,
)

DATA = ROOT / "data" / "naion_external"
OUT = ROOT / "results" / "naion_prefix_gate_v1"
SEEDS = (42, 43, 44, 45, 46)


def _find(columns, token):
    matches = [c for c in columns if token in str(c)]
    if len(matches) != 1:
        raise ValueError(f"expected one column containing {token!r}, found {matches}")
    return matches[0]


def load_series():
    series = {}
    for path in sorted(DATA.glob("270040-1-*.csv")):
        frame = pd.read_csv(path, encoding="gbk")
        cycle_col = _find(frame.columns, "循环号")
        cap_col = _find(frame.columns, "放电容量")
        cycle = pd.to_numeric(frame[cycle_col], errors="coerce")
        cap = pd.to_numeric(frame[cap_col].replace("-", np.nan), errors="coerce")
        clean = pd.DataFrame({"cycle": cycle, "capacity": cap}).dropna()
        clean = clean[(clean.cycle >= 0) & (clean.capacity > 0)]
        grouped = clean.groupby("cycle", sort=True).capacity.max()
        t = grouped.index.to_numpy(float)
        c = grouped.to_numpy(float)
        if len(t) >= 40:
            series[path.name] = {"cycle": t, "capacity": c}
    return series


def features_at(t, c, i, time_scale):
    h = c / max(c[0], 1e-8)
    elapsed = (t - t[0]) / max(time_scale, 1e-8)
    slopes = [
        (h[i] - h[max(0, i - w)]) /
        (elapsed[i] - elapsed[max(0, i - w)] + 1e-8)
        for w in (1, 2, 3)
    ]
    hist = h[max(0, i - 3): i + 1]
    return [
        h[i], elapsed[i], *slopes, float(hist.mean()), float(hist.std()),
        float(h[i] - h[0]), float(np.min(h[: i + 1])), elapsed[i], 1.0,
    ]


def make_rows(series, ids, boundary, time_scale, side):
    xx, yy, gg = [], [], []
    for uid in ids:
        t = series[uid]["cycle"]
        c = series[uid]["capacity"]
        span = t - t[0]
        mask = span <= boundary if side == "prefix" else span > boundary
        for i in np.flatnonzero(mask & (np.arange(len(t)) >= 3)):
            xx.append(features_at(t, c, i, time_scale))
            yy.append(t[-1] - t[i])
            gg.append(uid)
    return {
        "x": np.asarray(xx, np.float32),
        "y": np.asarray(yy, np.float32),
        "groups": np.asarray(gg),
    }


def prefix_descriptor(t, c, boundary):
    use = np.flatnonzero((t - t[0]) <= boundary)
    if len(use) < 8:
        raise ValueError("insufficient prefix")
    tt = (t[use] - t[use][0]) / max(boundary, 1e-8)
    hh = c[use] / max(c[0], 1e-8)
    p = np.polyfit(tt, hh, 2)
    cut = max(4, len(tt) // 3)
    early = np.polyfit(tt[:cut], hh[:cut], 1)[0]
    late = np.polyfit(tt[-cut:], hh[-cut:], 1)[0]
    residual = hh - np.polyval(p, tt)
    return np.asarray([hh[-1], np.polyfit(tt, hh, 1)[0], early, late,
                       p[0], np.std(residual)], float)


def compatibility(train_desc, test_desc):
    center = np.median(train_desc, axis=0)
    q25, q75 = np.quantile(train_desc, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, np.maximum(np.abs(center) * 0.05, 1e-4))
    ztrain = (train_desc - center) / scale
    ztest = (test_desc - center) / scale
    return np.min(np.linalg.norm(ztest[:, None] - ztrain[None], axis=2), axis=1) / np.sqrt(train_desc.shape[1])


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
    aff = select_affine_initialization(tr, va)
    plain_models, pp_models, pval, qval = [], [], [], []
    for seed in SEEDS:
        fm = fit_plain(tr, va, seed=seed)
        fp = fit_pp(tr, va, seed=seed, affine_selection=aff)
        plain_models.append(fm)
        pp_models.append(fp)
        pval.append(predict_plain(fm, va["x"]))
        qval.append(predict(fp, va["x"]))
    pval, qval = np.asarray(pval), np.asarray(qval)
    pve, qve = pval.mean(0), qval.mean(0)
    pvm = regression_metrics(va["y"], pve, va["groups"])
    qvm = regression_metrics(va["y"], qve, va["groups"])
    relative_gain = 1.0 - qvm["pooled"]["rmse"] ** 2 / max(pvm["pooled"]["rmse"] ** 2, 1e-12)
    disagreement = float(np.mean(np.std(qval, axis=0)) / max(np.std(va["y"]), 1e-8))

    train_desc = np.asarray([prefix_descriptor(series[u]["cycle"], series[u]["capacity"], boundary) for u in ids["train"]])
    test_desc = np.asarray([prefix_descriptor(series[u]["cycle"], series[u]["capacity"], boundary) for u in ids["test"]])
    distances = compatibility(train_desc, test_desc)
    checks = {
        "positive_validation_pp_r2": bool(qvm["pooled"]["r2"] > 0),
        "validation_mse_gain_at_least_2pct": bool(relative_gain >= 0.02),
        "seed_disagreement_at_most_0_25": bool(disagreement <= 0.25),
        "enough_units": bool(len(ids["train"]) >= 4 and len(ids["validation"]) >= 2),
        "all_prefix_distances_at_most_3": bool(np.all(distances <= 3.0)),
    }
    commitment = hashlib.sha256(json.dumps({
        "test_ids": ids["test"], "test_prefix_descriptors": test_desc.tolist()
    }, sort_keys=True).encode()).hexdigest()
    gate = {
        "protocol": "protocols/NAION_PREFIX_GATE_PROTOCOL.md",
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
    gate_path = OUT / "gate_decision_preoutcome.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")

    # Outcome materialization starts only after the immutable pre-outcome artifact.
    te = make_rows(series, ids["test"], boundary, time_scale, "tail")
    ptest = np.asarray([predict_plain(m, te["x"]) for m in plain_models])
    qtest = np.asarray([predict(m, te["x"]) for m in pp_models])
    pm = regression_metrics(te["y"], ptest.mean(0), te["groups"])
    qm = regression_metrics(te["y"], qtest.mean(0), te["groups"])
    actual_success = bool(qm["pooled"]["r2"] > 0 and qm["pooled"]["r2"] >= pm["pooled"]["r2"])
    result = {
        "status": "prospective one-shot external evaluation",
        "gate": gate,
        "n": {"train": len(tr["y"]), "validation": len(va["y"]), "test": len(te["y"])},
        "plain_ensemble": pm,
        "pp_ensemble": qm,
        "pp_gain_r2": qm["pooled"]["r2"] - pm["pooled"]["r2"],
        "actual_pp_success": actual_success,
        "gate_correct": bool(gate["approved"] == actual_success),
        "selective_outcome": "PP" if gate["approved"] else "ABSTAIN",
        "seed_runs": [
            {"seed": seed,
             "plain_r2": regression_metrics(te["y"], ptest[i], te["groups"])["pooled"]["r2"],
             "pp_r2": regression_metrics(te["y"], qtest[i], te["groups"])["pooled"]["r2"]}
            for i, seed in enumerate(SEEDS)
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=te["y"], groups=te["groups"], plain=ptest, pp=qtest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
