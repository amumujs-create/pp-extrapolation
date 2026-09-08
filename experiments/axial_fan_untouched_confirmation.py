#!/usr/bin/env python3
"""Locked confirmation on three outcome-sealed axial-fan RUL configurations."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, select_affine_initialization
from pp_extrapolation.model import transform_features

DATA = ROOT / "data" / "axial_fan"
OUT = ROOT / "results" / "axial_fan_three_config_confirmation_v1"
CONFIGS = {"1P_8F": 1, "4P_1F": 4, "4P_8F": 4}
SEEDS = (42, 43, 44, 45, 46)
WINDOW, INITIAL = 30, 20


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Normalizer:
    op_center: np.ndarray
    op_scale: np.ndarray
    centers: np.ndarray
    health_center: np.ndarray
    health_scale: np.ndarray

    @classmethod
    def fit(cls, table, n_regimes):
        op, health = table[:, 2:5], table[:, 5:13]
        oc, oscale = op.mean(0), op.std(0)
        oscale[oscale < 1e-8] = 1.0
        zop = (op - oc) / oscale
        if n_regimes == 1:
            centers = np.zeros((1, 3))
            labels = np.zeros(len(table), dtype=int)
        else:
            raw = KMeans(n_clusters=n_regimes, n_init=20, random_state=42).fit(zop).cluster_centers_
            order = np.lexsort((raw[:, 2], raw[:, 1], raw[:, 0]))
            centers = raw[order]
            labels = np.argmin(((zop[:, None] - centers[None]) ** 2).sum(2), axis=1)
        hc, hs = [], []
        for k in range(n_regimes):
            block = health[labels == k]
            center, scale = block.mean(0), block.std(0)
            scale[scale < 1e-8] = 1.0
            hc.append(center); hs.append(scale)
        return cls(oc, oscale, centers, np.asarray(hc), np.asarray(hs))

    def transform(self, table):
        zop = (table[:, 2:5] - self.op_center) / self.op_scale
        labels = np.argmin(((zop[:, None] - self.centers[None]) ** 2).sum(2), axis=1)
        zhealth = (table[:, 5:13] - self.health_center[labels]) / self.health_scale[labels]
        occupancy = np.zeros((len(table), 4)); occupancy[np.arange(len(table)), labels] = 1.0
        return np.column_stack((zhealth, zop, occupancy))


def split_units(table):
    units = np.unique(table[:, 0].astype(int))
    rng = np.random.default_rng(42)
    val = np.sort(rng.choice(units, 20, replace=False))
    mask = np.isin(table[:, 0].astype(int), val)
    return table[~mask], table[mask], units[~np.isin(units, val)], val


def feature_at(group, z, endpoint):
    start = max(0, endpoint - WINDOW + 1)
    block = z[start:endpoint + 1]
    if len(block) < WINDOW:
        block = np.vstack((np.repeat(block[:1], WINDOW-len(block), 0), block))
    health = block[:, :8]
    time = np.arange(WINDOW, dtype=float); time -= time.mean()
    slope = time @ health / np.sum(time * time)
    initial = z[:min(INITIAL, endpoint + 1), :8].mean(0)
    return np.concatenate((health[-1], health.mean(0), health.std(0), slope,
                           health[-1]-initial, block[-1, 8:11], block[:, 11:].mean(0),
                           [np.log1p(group[endpoint, 1])]))


def rows(table, normalizer, *, final_only=False):
    z = normalizer.transform(table)
    xs, ys, groups = [], [], []
    for unit in sorted(np.unique(table[:, 0].astype(int))):
        ix = np.flatnonzero(table[:, 0].astype(int) == unit)
        ix = ix[np.argsort(table[ix, 1])]
        group, zg = table[ix], z[ix]
        endpoints = [len(group)-1] if final_only else list(range(min(WINDOW-1, len(group)-1), len(group), 5))
        if endpoints[-1] != len(group)-1: endpoints.append(len(group)-1)
        for endpoint in endpoints:
            xs.append(feature_at(group, zg, endpoint))
            ys.append(group[-1, 1] - group[endpoint, 1])
            groups.append(str(unit))
    return {"x": np.asarray(xs, np.float32), "y": np.asarray(ys), "groups": np.asarray(groups)}


def load_features(tag, n_regimes):
    train = np.loadtxt(DATA / f"train_FAN_{tag}.txt")
    test = np.loadtxt(DATA / f"test_FAN_{tag}.txt")
    if train.shape[1] != 13 or test.shape[1] != 13:
        raise RuntimeError(f"{tag}: corrected 13-column schema failed")
    fit_table, val_table, fit_units, val_units = split_units(train)
    sn = Normalizer.fit(fit_table, n_regimes)
    fit, val = rows(fit_table, sn), rows(val_table, sn)
    fn = Normalizer.fit(train, n_regimes)
    full, endpoint = rows(train, fn), rows(test, fn, final_only=True)
    # Unit-level exact duplicates are forbidden across train and test.
    signatures = lambda t: {hashlib.sha256(np.ascontiguousarray(t[t[:, 0] == u, 1:]).tobytes()).hexdigest()
                            for u in np.unique(t[:, 0])}
    if signatures(train) & signatures(test):
        raise RuntimeError(f"{tag}: duplicate train/test unit trajectory")
    return fit, val, full, endpoint, fit_units, val_units


def affine_prediction(fit, x):
    value = torch.as_tensor(transform_features(x, fit.center, fit.scale), dtype=torch.float32)
    fit.model.eval()
    with torch.no_grad():
        out = fit.model.affine(value).squeeze(1).numpy() * fit.target_scale
    return np.clip(out, 0.0, fit.target_scale)


def metric(y, p):
    return {"r2": float(r2_score(y, p)), "rmse": float(mean_squared_error(y, p)**0.5),
            "mae": float(mean_absolute_error(y, p))}


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    frozen, audit = {}, {}
    for tag, n_regimes in CONFIGS.items():
        train, val, full, endpoint, fit_units, val_units = load_features(tag, n_regimes)
        selection = select_affine_initialization(train, val)
        pp, mlp, affine, epochs = [], [], [], []
        for seed in SEEDS:
            sf = fit_pp(train, val, seed=seed, affine_selection=selection, max_epochs=400, patience=70)
            pe = max(int(sf.selection["selected_epoch"]), 1)
            final_selection = select_affine_initialization(full, full, alphas=(selection["selected_alpha"],))
            pf = fit_pp(full, full, seed=seed, affine_selection=final_selection,
                        max_epochs=pe, patience=pe+1)
            sm = fit_plain(train, val, seed=seed, max_epochs=400, patience=70)
            me = max(int(sm["selected_epoch"]), 1)
            mf = fit_plain(full, full, seed=seed, max_epochs=me, patience=me+1, restore_best=False)
            pp.append(predict(pf, endpoint["x"])); mlp.append(predict_plain(mf, endpoint["x"]))
            affine.append(affine_prediction(pf, endpoint["x"])); epochs.append([pe, me])
            print("PREDICTION_FROZEN", tag, seed, pe, me, flush=True)
        frozen[tag] = {"pp": np.asarray(pp), "mlp": np.asarray(mlp), "affine": np.asarray(affine),
                       "groups": endpoint["groups"], "epochs": epochs}
        audit[tag] = {"fit_units": fit_units.tolist(), "validation_units": val_units.tolist(),
                      "n_features": int(endpoint["x"].shape[1]), "n_test_units": len(endpoint["x"])}

    # Durable timestamp-free artifact proves every prediction existed before labels were parsed.
    np.savez_compressed(OUT / "predictions_frozen_before_labels.npz",
        **{f"{tag}_{key}": value[key] for tag, value in frozen.items() for key in ("pp", "mlp", "affine", "groups")})

    results, y_pool, pp_pool, mlp_pool = {}, [], [], []
    for tag in CONFIGS:
        truth_path = DATA / "sealed" / f"RUL_FAN_{tag}.txt"
        truth = np.loadtxt(truth_path).reshape(-1)
        if len(truth) != len(frozen[tag]["groups"]): raise RuntimeError(f"{tag}: truth count mismatch")
        p, m, a = frozen[tag]["pp"].mean(0), frozen[tag]["mlp"].mean(0), frozen[tag]["affine"].mean(0)
        results[tag] = {"pp": metric(truth, p), "plain_mlp": metric(truth, m),
                        "affine": metric(truth, a), "runs": [
                            {"seed": s, "pp": metric(truth, frozen[tag]["pp"][i]),
                             "plain_mlp": metric(truth, frozen[tag]["mlp"][i]),
                             "pp_epoch": frozen[tag]["epochs"][i][0], "mlp_epoch": frozen[tag]["epochs"][i][1]}
                            for i, s in enumerate(SEEDS)]}
        y_pool.append(truth); pp_pool.append(p); mlp_pool.append(m)
        print("SCORED", tag, results[tag]["pp"], results[tag]["plain_mlp"], flush=True)

    y, p, m = map(np.concatenate, (y_pool, pp_pool, mlp_pool))
    rng = np.random.default_rng(20260909); n = len(y)
    index = rng.integers(0, n, size=(20000, n))
    gain = np.sqrt(np.mean((m[index]-y[index])**2, 1))-np.sqrt(np.mean((p[index]-y[index])**2, 1))
    macro_pp = float(np.mean([results[t]["pp"]["r2"] for t in CONFIGS]))
    macro_mlp = float(np.mean([results[t]["plain_mlp"]["r2"] for t in CONFIGS]))
    wins = sum(results[t]["pp"]["rmse"] < results[t]["plain_mlp"]["rmse"] for t in CONFIGS)
    pooled = {"pp": metric(y, p), "plain_mlp": metric(y, m)}
    inference = {"rmse_gain_mlp_minus_pp": pooled["plain_mlp"]["rmse"]-pooled["pp"]["rmse"],
                 "bootstrap_95ci": [float(x) for x in np.quantile(gain, (.025, .975))],
                 "bootstrap_probability_positive": float(np.mean(gain > 0)), "replicates": 20000}
    success = bool(pooled["pp"]["r2"] > 0 and macro_pp > macro_mlp and wins >= 2 and
                   inference["bootstrap_probability_positive"] >= .95)
    payload = {"status": "untouched confirmation complete", "protocol_commit": "1c32b60",
               "dataset_doi": "10.17632/mzjvw6kbt7.1", "integrity": audit,
               "file_sha256": {path.name: sha256(path) for path in sorted(DATA.glob("*.txt"))}
                              | {path.name: sha256(path) for path in sorted((DATA/"sealed").glob("*.txt"))},
               "results": results, "aggregate": {"macro_r2": {"pp": macro_pp, "plain_mlp": macro_mlp},
               "pooled": pooled, "pp_rmse_wins": wins, "paired_inference": inference},
               "confirmatory_success": success,
               "claim_scope": "unseen-fan truncated-future endpoint; strict hull status not asserted"}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], indent=2), "SUCCESS", success)


if __name__ == "__main__":
    main()
