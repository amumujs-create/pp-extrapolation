#!/usr/bin/env python3
"""Retrospective same-information ML benchmark on principal CCMR cohorts."""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pyreadr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cohort_ml_benchmark"


def module(name):
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def subset(rows, chosen=None):
    if chosen is None:
        chosen = np.ones(len(rows["y"]), dtype=bool)
    return {
        "x": np.asarray(rows["context"][chosen], dtype=float),
        "y": np.asarray(rows["y"][chosen], dtype=float),
        "groups": np.asarray(rows["groups"][chosen]),
    }


def load_cohorts():
    cohorts = {}

    alloy = module("alloya_unopened_ccmr_v16")
    frame = next(iter(pyreadr.read_r(str(alloy.DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    split = alloy.frozen_split(
        sorted(frame["specimen"].astype(str).unique())
    )
    cohorts["Alloy_A"] = {
        mode: subset(alloy.make_rows(frame, split[mode], mode))
        for mode in ("train", "validation", "test")
    }

    concrete = module("concrete_material_shift_ccmr_v17")
    trajectories = {
        mode: concrete.load_trajectories(
            concrete.SPLIT_FILES[mode]
        )
        for mode in ("train", "validation", "test")
    }
    concrete_test = concrete.make_rows(trajectories["test"])
    cohorts["Concrete_HPC_to_UHPC"] = {
        "train": subset(concrete.make_rows(
            trajectories["train"], (0.0, 0.30)
        )),
        "validation": subset(concrete.make_rows(
            trajectories["validation"], (0.40, 0.60)
        )),
        "test": subset(
            concrete_test,
            (concrete_test["progress"] >= 0.75)
            & (concrete_test["progress"] <= 0.90),
        ),
    }

    sit = module("sit_lfp_sequential_ccmr_v17")
    trajectories = sit.load_trajectories()
    split = sit.split_units(trajectories)
    cohorts["SIT_LFP"] = {
        "train": subset(sit.make_rows(
            trajectories, split["train"], (0.0, 0.30)
        )),
        "validation": subset(sit.make_rows(
            trajectories, split["validation"], (0.40, 0.60)
        )),
        "test": subset(sit.make_rows(
            trajectories, split["test"], (0.75, 0.90)
        )),
    }

    radar = module("radar_nmc_cyclic_ccmr_v18")
    trajectories, _ = radar.load_trajectories()
    split = radar.split_units(trajectories)
    cohorts["RADAR_NMC"] = {
        "train": subset(radar.make_rows(
            trajectories, split["train"], (0.0, 0.30)
        )),
        "validation": subset(radar.make_rows(
            trajectories, split["validation"], (0.40, 0.60)
        )),
        "test": subset(radar.make_rows(
            trajectories, split["test"], (0.75, 0.90)
        )),
    }

    multi = module("multistage_rpt_ccmr_v19")
    trajectories = multi.load_trajectories()
    split = multi.split_units(trajectories)
    cohorts["MultiStage_RPT"] = {
        "train": subset(multi.make_rows(
            trajectories, split["train"], (0.0, 0.30)
        )),
        "validation": subset(multi.make_rows(
            trajectories, split["validation"], (0.40, 0.60)
        )),
        "test": subset(multi.make_rows(
            trajectories, split["test"], (0.75, 0.90)
        )),
    }
    return cohorts


def group_weights(groups):
    _, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    value = 1.0 / counts[inverse]
    return value / np.mean(value)


def candidates():
    return {
        "linear": [
            make_pipeline(StandardScaler(), Ridge(alpha=1e-8))
        ],
        "ridge": [
            make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)
        ],
        "random_forest": [
            RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=leaf,
                max_features=0.8,
                random_state=19,
                n_jobs=-1,
            )
            for leaf in (1, 5, 20)
        ],
        "extra_trees": [
            ExtraTreesRegressor(
                n_estimators=150,
                min_samples_leaf=leaf,
                max_features=0.8,
                random_state=19,
                n_jobs=-1,
            )
            for leaf in (1, 5, 20)
        ],
        "hist_gradient_boosting": [
            HistGradientBoostingRegressor(
                max_iter=250,
                max_leaf_nodes=leaves,
                l2_regularization=l2,
                random_state=19,
            )
            for leaves in (7, 15)
            for l2 in (1.0, 10.0)
        ],
        "mlp": [
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    alpha=alpha,
                    learning_rate_init=1e-3,
                    max_iter=1500,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=80,
                    random_state=19,
                ),
            )
            for hidden in ((16,), (32, 16))
            for alpha in (0.01, 0.1, 1.0)
        ],
    }


def fit_candidate(model, train):
    weights = group_weights(train["groups"])
    final = model.steps[-1][0] if hasattr(model, "steps") else None
    if final == "mlpregressor":
        model.fit(train["x"], train["y"])
    elif hasattr(model, "steps"):
        model.fit(
            train["x"], train["y"],
            **{f"{final}__sample_weight": weights},
        )
    else:
        model.fit(train["x"], train["y"], sample_weight=weights)
    return model


def clipped_predict(model, rows, train_y):
    lower, upper = np.quantile(train_y, [0.005, 0.995])
    span = max(float(upper - lower), 1e-6)
    return np.clip(
        model.predict(rows["x"]),
        lower - 0.2 * span,
        upper + 0.2 * span,
    )


def metrics(rows, prediction):
    anchor = rows["x"][:, 0]
    model = regression_metrics(rows["y"], prediction, rows["groups"])
    baseline = regression_metrics(rows["y"], anchor, rows["groups"])
    model_macro = np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ])
    baseline_macro = np.mean([
        value["rmse"] for value in baseline["per_unit"].values()
    ])
    raw = regret_summary(raw_unit_regret(
        rows["y"], rows["groups"], anchor, prediction
    ))
    return {
        "r2": model["pooled"]["r2"],
        "rmse": model["pooled"]["rmse"],
        "pooled_improvement": (
            baseline["pooled"]["rmse"] - model["pooled"]["rmse"]
        ) / baseline["pooled"]["rmse"],
        "macro_improvement": (
            baseline_macro - model_macro
        ) / baseline_macro,
        "raw_mean_regret": raw[0],
        "raw_cvar20_regret": raw[1],
        "raw_max_regret": raw[2],
    }


SEALED = {
    "Alloy_A": (
        "results/alloya_unopened_ccmr_v16/sealed_predictions.npz",
        "prediction",
    ),
    "Concrete_HPC_to_UHPC": (
        "results/concrete_material_shift_ccmr_v17/sealed_predictions.npz",
        "deployed",
    ),
    "SIT_LFP": (
        "results/sit_lfp_sequential_ccmr_v17_amended/sealed_predictions.npz",
        "deployed",
    ),
    "RADAR_NMC": (
        "results/radar_nmc_cyclic_ccmr_v18/sealed_predictions.npz",
        "deployed",
    ),
    "MultiStage_RPT": (
        "results/multistage_rpt_ccmr_v19_amended2/sealed_predictions.npz",
        "deployed",
    ),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "results_v2.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite ML benchmark")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cohorts = load_cohorts()
    result = {
        "status": "retrospective opened-test ML benchmark",
        "comparison_is_confirmatory": False,
        "cohorts": {},
    }
    for cohort_name, rows in cohorts.items():
        cohort_result = {
            "rows": {
                key: len(value["y"]) for key, value in rows.items()
            },
            "models": {},
        }
        anchor = rows["test"]["x"][:, 0]
        cohort_result["models"]["persistence"] = metrics(
            rows["test"], anchor
        )
        sealed_path, prediction_key = SEALED[cohort_name]
        sealed = np.load(ROOT / sealed_path)
        assert len(sealed["truth"]) == len(rows["test"]["y"])
        assert np.allclose(sealed["truth"], rows["test"]["y"])
        cohort_result["models"]["CCMR_deployed"] = metrics(
            rows["test"], sealed[prediction_key]
        )
        if cohort_name == "RADAR_NMC":
            cohort_result["models"]["CCMR_raw_candidate"] = metrics(
                rows["test"], sealed["candidate"]
            )
        for representation in ("direct", "residual"):
            fitting_rows = rows["train"]
            if representation == "residual":
                fitting_rows = {
                    **rows["train"],
                    "y": (
                        rows["train"]["y"]
                        - rows["train"]["x"][:, 0]
                    ),
                }
            for family, family_candidates in candidates().items():
                best = None
                for index, candidate in enumerate(family_candidates):
                    fitted = fit_candidate(candidate, fitting_rows)
                    validation_prediction = clipped_predict(
                        fitted,
                        rows["validation"],
                        fitting_rows["y"],
                    )
                    if representation == "residual":
                        validation_prediction = (
                            rows["validation"]["x"][:, 0]
                            + validation_prediction
                        )
                    validation_rmse = regression_metrics(
                        rows["validation"]["y"],
                        validation_prediction,
                        rows["validation"]["groups"],
                    )["pooled"]["rmse"]
                    if best is None or validation_rmse < best[0]:
                        best = (validation_rmse, index, fitted)
                test_prediction = clipped_predict(
                    best[2], rows["test"], fitting_rows["y"]
                )
                if representation == "residual":
                    test_prediction = (
                        rows["test"]["x"][:, 0] + test_prediction
                    )
                key = f"{representation}_{family}"
                cohort_result["models"][key] = {
                    "selected_candidate_index": best[1],
                    "validation_rmse": best[0],
                    **metrics(rows["test"], test_prediction),
                }
        result["cohorts"][cohort_name] = cohort_result
        print(cohort_name, flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
