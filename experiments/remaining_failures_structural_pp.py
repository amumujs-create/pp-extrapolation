#!/usr/bin/env python3
"""Development routes for the observed XJTU and FEMTO failure settings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]

from pp_extrapolation import (capacity_from_independent_groups, fit_pp, predict,
                              progress_quotient, regression_metrics,
                              select_affine_initialization)

OUT = ROOT / "results" / "remaining_failures_structural_pp_v1"
SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)


def positions(rows):
    value = np.zeros(len(rows["y"]), dtype=np.float64)
    for unit in np.unique(rows["groups"]):
        idx = np.flatnonzero(rows["groups"] == unit)
        value[idx] = np.arange(8, 8 + len(idx))
    return value


def xjtu_progress_route():
    from xjtu_untouched import build_cache, windows
    raw = build_cache()
    train = windows(raw, "37.5Hz11kN")
    validation = windows(raw, "35Hz12kN")
    test = windows(raw, "40Hz10kN")
    pt, pv, ps = map(positions, (train, validation, test))

    def augment(rows, pos):
        return np.column_stack([rows["x"], pos, np.log1p(pos)]).astype(np.float64)

    x, v, s = augment(train, pt), augment(validation, pv), augment(test, ps)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.0
    x, v, s = ((a-center)/scale for a in (x, v, s))
    progress = np.clip(pt / (pt + train["y"]), 1e-4, 1-1e-4)
    logit_progress = np.log(progress / (1-progress))
    search = []
    for alpha in (.01, .1, 1., 10., 100., 1000., 10000., 100000.):
        model = Ridge(alpha=alpha).fit(x, logit_progress)
        # RUL = elapsed * odds(remaining) = elapsed * exp(-logit(progress)).
        pred = progress_quotient(pv, model.predict(v))
        search.append({"alpha": alpha, "validation_rmse": float(np.sqrt(np.mean((pred-validation["y"])**2)))})
    selected = min(search, key=lambda row: row["validation_rmse"])
    model = Ridge(alpha=selected["alpha"]).fit(x, logit_progress)
    prediction = progress_quotient(ps, model.predict(s))
    return {
        "model": "scale-free progress-quotient PP affine head",
        "formula": "RUL=elapsed*exp(-logit_progress)",
        "search": search,
        "selected_alpha": selected["alpha"],
        "test": regression_metrics(test["y"], prediction, test["groups"]),
    }, (test, prediction)


def femto_features(frame, sensor_columns):
    result = {}
    for _, unit in frame.groupby("bearing", sort=False):
        unit = unit.sort_values("cycle")
        values = unit[sensor_columns].to_numpy(np.float64)
        baseline = np.median(values[:min(10, len(values))], axis=0)
        scale = np.maximum(np.abs(baseline), 1e-5)
        for j, (idx, row) in enumerate(unit.iterrows()):
            j8, j32 = max(0, j-7), max(0, j-31)
            recent = values[max(0, j-15):j+1]
            result[idx] = np.r_[
                row.condition, row.cycle/1000., np.log1p(row.cycle)/8., values[j],
                (values[j]-baseline)/scale,
                (values[j]-values[j8])/max(j-j8, 1)/scale,
                (values[j]-values[j32])/max(j-j32, 1)/scale,
                np.std(recent, axis=0)/scale,
                np.max(recent, axis=0)/scale,
            ].astype(np.float32)
    return result


def femto_prefix_route():
    from femto_bearing_loader import FEATURE_COLS, load_femto_phm2012
    frame, groups = load_femto_phm2012(root=ROOT / "data/femto/raw", file_stride=5)
    feature_map = femto_features(frame, FEATURE_COLS[1:])

    def rows(units, endpoint=False):
        part = frame[frame.unit.isin(units)].copy()
        if endpoint:
            part = part.sort_values("cycle").groupby("unit", as_index=False).tail(1)
        return {"x": np.asarray([feature_map[i] for i in part.index], np.float32),
                "y": part.RUL.to_numpy(np.float32), "groups": part.bearing.to_numpy()}

    train, validation, test = rows(groups["train"]), rows(groups["val"]), rows(groups["test"], True)
    # Capacity is determined from training group count, not validation/test
    # performance. Five independent trajectories permit an 8-wide residual.
    width = capacity_from_independent_groups(len(np.unique(train["groups"])))
    affine = select_affine_initialization(train, validation)
    search = []
    for lr in (1e-4, 2e-4, 5e-4, 1e-3):
        for wd in (.1, 2., 10.):
            vp = []
            for seed in SELECTION_SEEDS:
                fit = fit_pp(train, validation, seed=seed, affine_selection=affine,
                    width=width, learning_rate=lr, weight_decay=wd,
                    direct_residual_mixture=True, fixed_affine_trust=0.,
                    residual_seed_replay=True, residual_zero_init=False)
                vp.append(predict(fit, validation["x"]))
            prediction = np.mean(vp, axis=0)
            search.append({"learning_rate": lr, "weight_decay": wd,
                "validation_rmse": float(np.sqrt(np.mean((prediction-validation["y"])**2)))})
    selected = min(search, key=lambda row: row["validation_rmse"])
    predictions = []
    for seed in FINAL_SEEDS:
        fit = fit_pp(train, validation, seed=seed, affine_selection=affine,
            width=width, learning_rate=selected["learning_rate"], weight_decay=selected["weight_decay"],
            direct_residual_mixture=True, fixed_affine_trust=0.,
            residual_seed_replay=True, residual_zero_init=False)
        predictions.append(predict(fit, test["x"]))
    prediction = np.mean(predictions, axis=0)
    return {
        "model": "unit-count capacity-controlled causal-prefix PP safety route",
        "width_rule": "next power of two >= number of independent training units",
        "selected_width": width,
        "search": search,
        "selected": selected,
        "test": regression_metrics(test["y"], prediction, test["groups"]),
    }, (test, np.asarray(predictions))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    xjtu, (xt, xp) = xjtu_progress_route()
    femto, (ft, fp) = femto_prefix_route()
    payload = {"status": "post-test structural development; no new cohort opened",
               "xjtu": xjtu, "femto": femto}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "xjtu_predictions.npz", y=xt["y"], groups=xt["groups"], prediction=xp)
    np.savez_compressed(OUT / "femto_predictions.npz", y=ft["y"], groups=ft["groups"], prediction=fp)
    print(json.dumps({"xjtu": xjtu["test"]["pooled"], "femto": femto["test"]["pooled"],
                      "femto_selected": femto["selected"]}, indent=2))


if __name__ == "__main__":
    main()
