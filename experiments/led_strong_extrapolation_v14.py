#!/usr/bin/env python3
"""Strong age extrapolation on three compact external LED cohorts."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.regime_prior_bank import (
    fit_auto_regime_prior,
    predict_auto_regime_prior,
)
from pp_extrapolation.risk_budgeted_prior import (
    fit_support_scale,
    support_distance,
)
from pp_extrapolation.stability_first import regret_summary, unit_regret
from stanford_ppx_v11_safety import fit_route

DATA = ROOT / "data/led_external_v14"
OUT = ROOT / "results/led_strong_extrapolation_v14"
DATA_IDS = (2, 5, 6)
HISTORY = 5
HORIZON = 2
CONFIGS = (
    {"width": 16, "learning_rate": 1e-3, "weight_decay": 2.0},
    {"width": 32, "learning_rate": 1e-3, "weight_decay": 2.0},
)
TRUSTS = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.40])
SEEDS = (42, 43, 44, 45, 46)


def load_cells(data_id):
    data = pd.read_csv(DATA / f"Data_ID_{data_id}.csv")
    cells = {}
    for unit, frame in data.groupby(["Test_ID", "Sample_ID"]):
        test_id, sample = unit
        frame = frame.sort_values("Time", kind="stable")
        time = frame["Time"].to_numpy(float)
        flux = frame["Flux_t"].to_numpy(float)
        finite = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[finite], flux[finite]
        unique_time = np.unique(time)
        health = np.asarray([
            np.median(flux[time == value]) for value in unique_time
        ])
        scale = float(np.median(health[:3]))
        if len(unique_time) >= HISTORY + HORIZON + 1 and scale > 0:
            cells[f"D{data_id}_T{test_id}_S{sample}"] = {
                "time": unique_time,
                "health": health / scale,
            }
    return cells


def split_ids(cells, data_id):
    ids = np.asarray(sorted(cells))
    order = np.random.default_rng(20260910 + data_id).permutation(ids)
    first = int(0.6 * len(order))
    second = int(0.8 * len(order))
    return {
        "train": list(order[:first]),
        "validation": list(order[first:second]),
        "test": list(order[second:]),
    }


def train_time_scale(cells, names):
    maxima = [
        cells[name]["time"][len(cells[name]["time"]) - HORIZON - 1]
        for name in names
    ]
    return float(np.quantile(maxima, 0.90))


def make_rows(cells, names, time_scale, mode):
    limits = {
        "train": (0.0, 0.30),
        "validation": (0.40, 0.60),
        "test": (0.75, 1.0),
    }
    lower, upper = limits[mode]
    x, y, groups, age, progress_values, horizon_delta = [], [], [], [], [], []
    for name in names:
        time = cells[name]["time"]
        health = cells[name]["health"]
        origins = list(range(HISTORY - 1, len(time) - HORIZON))
        denominator = max(len(origins) - 1, 1)
        for origin_position, index in enumerate(origins):
            progress = float(origin_position / denominator)
            normalized_time = float(time[index] / time_scale)
            if not lower <= progress <= upper:
                continue
            window = health[index - HISTORY + 1:index + 1]
            slopes = []
            for lag in (1, 3, 4):
                start = max(0, index - lag)
                dt = max(float(time[index] - time[start]), 1e-6)
                slopes.append(float((health[index] - health[start]) / dt))
            x.append([
                float(health[index]),
                *slopes,
                float(window.mean()),
                float(window.std()),
                normalized_time,
                float(min((index + 1) / HISTORY, 1.0)),
            ])
            y.append(float(health[index + HORIZON]))
            groups.append(name)
            age.append(normalized_time)
            progress_values.append(progress)
            horizon_delta.append(float(time[index + HORIZON] - time[index]))
    return {
        "x": np.asarray(x, np.float32),
        "y": np.asarray(y, np.float32),
        "groups": np.asarray(groups),
        "age": np.asarray(age),
        "progress": np.asarray(progress_values),
        "horizon_delta": np.asarray(horizon_delta),
    }


def score(split, prediction, baseline):
    regret = regret_summary(unit_regret(
        split["y"], split["groups"], baseline, prediction
    ))
    return {
        "model": regression_metrics(
            split["y"], prediction, split["groups"]
        ),
        "baseline": regression_metrics(
            split["y"], baseline, split["groups"]
        ),
        "regret": {
            "mean": regret[0], "cvar20": regret[1], "maximum": regret[2],
        },
    }


def evaluate(data_id):
    cells = load_cells(data_id)
    ids = split_ids(cells, data_id)
    time_scale = train_time_scale(cells, ids["train"])
    train = make_rows(cells, ids["train"], time_scale, "train")
    validation = make_rows(
        cells, ids["validation"], time_scale, "validation"
    )
    test = make_rows(cells, ids["test"], time_scale, "test")
    if len(np.unique(test["groups"])) < 4 or len(test["y"]) < 20:
        return {
            "status": "inconclusive",
            "reason": "fewer than four test units or 20 test rows",
            "split_ids": ids,
            "rows": {
                "train": len(train["y"]),
                "validation": len(validation["y"]),
                "test": len(test["y"]),
            },
        }, {}

    validation_members = {float(trust): [] for trust in TRUSTS}
    test_members = {float(trust): [] for trust in TRUSTS}
    for config in CONFIGS:
        for trust in TRUSTS:
            val, tst, _ = fit_route(
                train, validation, test, config, trust, SEEDS
            )
            validation_members[float(trust)].append(val)
            test_members[float(trust)].append(tst)
    validation_portfolio = {
        trust: np.concatenate(members).mean(0)
        for trust, members in validation_members.items()
    }
    test_portfolio = {
        trust: np.concatenate(members).mean(0)
        for trust, members in test_members.items()
    }
    prior_trusts = TRUSTS[1:]
    baseline_val = validation_portfolio[0.0]
    baseline_test = test_portfolio[0.0]
    validation_priors = np.stack([
        validation_portfolio[float(trust)] for trust in prior_trusts
    ])
    test_priors = np.stack([
        test_portfolio[float(trust)] for trust in prior_trusts
    ])
    decision = fit_auto_regime_prior(
        train["x"], validation["x"], validation["y"], validation["groups"],
        baseline_val, validation_priors, prior_trusts,
        n_regimes=2, min_regime_groups=3,
    )
    prediction = predict_auto_regime_prior(
        decision, test["x"], baseline_test, test_priors, prior_trusts
    )

    # Contract-aware OOD: slopes (1:4) and local noise (5), not age/health.
    regime_columns = [1, 2, 3, 5]
    support = fit_support_scale(train["x"][:, regime_columns])
    validation_distance = support_distance(
        validation["x"][:, regime_columns], support
    )
    test_distance = support_distance(test["x"][:, regime_columns], support)
    distance_threshold = float(np.max(validation_distance))
    out_of_support = test_distance > distance_threshold
    guarded = np.where(out_of_support, baseline_test, prediction)
    metrics = score(test, guarded, baseline_test)
    pooled = metrics["model"]["pooled"]
    baseline_pooled = metrics["baseline"]["pooled"]
    regret = metrics["regret"]
    passed = bool(
        pooled["rmse"] <= baseline_pooled["rmse"]
        and regret["mean"] <= 0.02
        and regret["cvar20"] <= 0.05
        and regret["maximum"] <= 0.10
    )
    result = {
        "status": "complete",
        "split_ids": ids,
        "rows": {
            "train": len(train["y"]),
            "validation": len(validation["y"]),
            "test": len(test["y"]),
        },
        "time_scale": time_scale,
        "ordered_extrapolation": {
            "maximum_train_progress": float(np.max(train["progress"])),
            "minimum_validation_progress": float(np.min(
                validation["progress"]
            )),
            "minimum_test_progress": float(np.min(test["progress"])),
            "test_outside_train_fraction": float(np.mean(
                test["progress"] > np.max(train["progress"])
            )),
            "test_time_above_train_max_fraction": float(np.mean(
                test["age"] > np.max(train["age"])
            )),
        },
        "decision": {
            "common_scale": decision.common_scale,
            "accepted_regimes": decision.accepted_regimes,
            "regimes": [asdict(item) for item in decision.regime_decisions],
        },
        "ood_guard": {
            "validation_max_distance": distance_threshold,
            "test_out_of_support_rows": int(np.sum(out_of_support)),
            "test_out_of_support_fraction": float(np.mean(out_of_support)),
        },
        "test": metrics,
        "noninferiority_and_risk_pass": passed,
    }
    arrays = {
        "prediction": guarded,
        "ungarded_prediction": prediction,
        "baseline": baseline_test,
        "truth": test["y"],
        "groups": test["groups"],
        "age": test["age"],
        "progress": test["progress"],
        "ood": out_of_support,
    }
    return result, arrays


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite strong LED extrapolation")
    results, arrays = {}, {}
    for data_id in DATA_IDS:
        result, cohort_arrays = evaluate(data_id)
        results[f"Data_ID_{data_id}"] = result
        arrays.update({
            f"D{data_id}_{key}": value
            for key, value in cohort_arrays.items()
        })
        print(f"Data_ID_{data_id}", json.dumps({
            "status": result["status"],
            "rows": result["rows"],
            "test": result.get("test", {}).get("model", {}).get("pooled"),
            "baseline": result.get("test", {}).get(
                "baseline", {}
            ).get("pooled"),
            "regret": result.get("test", {}).get("regret"),
            "ood": result.get("ood_guard"),
        }), flush=True)
    completed = [
        result for result in results.values()
        if result["status"] == "complete"
    ]
    overall = bool(
        len(completed) == len(DATA_IDS)
        and all(
            result["ordered_extrapolation"][
                "test_outside_train_fraction"
            ] == 1.0
            for result in completed
        )
        and all(
            result["noninferiority_and_risk_pass"] for result in completed
        )
        and sum(
            result["test"]["model"]["pooled"]["r2"] > 0
            for result in completed
        ) >= 2
    )
    result_path.write_text(json.dumps({
        "status": "development complete",
        "protocol": "protocols/LED_STRONG_EXTRAPOLATION_V14_PROTOCOL.md",
        "selected_source_bytes": int(sum(
            (DATA / f"Data_ID_{data_id}.csv").stat().st_size
            for data_id in DATA_IDS
        )),
        "overall_success": overall,
        "cohorts": results,
    }, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    print("overall_success", overall, flush=True)


if __name__ == "__main__":
    main()
