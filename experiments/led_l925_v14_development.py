#!/usr/bin/env python3
"""Post-screen small LED cohort evaluation of auto-regime PP-X v1.4."""
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
    regime_probabilities,
)
from pp_extrapolation.risk_budgeted_prior import (
    fit_support_scale,
    support_distance,
)
from pp_extrapolation.stability_first import regret_summary, unit_regret
from stanford_ppx_v11_safety import fit_route

DATA = ROOT / "data/led_external_v14"
OUT = ROOT / "results/led_l925_v14_development"
THRESHOLD = 0.925
HISTORY = 5
CONFIGS = (
    {"width": 16, "learning_rate": 1e-3, "weight_decay": 2.0},
    {"width": 32, "learning_rate": 1e-3, "weight_decay": 2.0},
)
TRUSTS = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.40])
SEEDS = (42, 43, 44, 45, 46)


def load_file(number):
    data = pd.read_csv(DATA / f"Data_ID_{number}.csv")
    flux_name = "Flux_t" if "Flux_t" in data else "Flux"
    cells = {}
    unit_columns = (
        ["LED_type", "Sample_ID"]
        if "LED_type" in data else ["Sample_ID"]
    )
    for unit, frame in data.groupby(unit_columns):
        values = unit if isinstance(unit, tuple) else (unit,)
        sample = "_".join(str(value) for value in values)
        frame = frame.sort_values("Time", kind="stable")
        time = frame["Time"].to_numpy(float)
        flux = frame[flux_name].to_numpy(float)
        finite = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[finite], flux[finite]
        unique_time = np.unique(time)
        flux = np.asarray([
            np.median(flux[time == value]) for value in unique_time
        ])
        scale = float(np.median(flux[:3]))
        cells[f"D{number}_S{sample}"] = {
            "time": unique_time,
            "health": flux / scale,
            "source": f"Data_ID_{number}",
        }
    return cells


def crossing(cell):
    health = cell["health"]
    hits = np.flatnonzero(
        (health[:-1] <= THRESHOLD) & (health[1:] <= THRESHOLD)
    )
    value = int(hits[0]) if len(hits) else None
    return value if value is not None and len(health) >= 10 and value >= 8 else None


def make_rows(cells, names, crossings, *, boundary=None, train=False):
    x, y, groups, health, source = [], [], [], [], []
    for name in names:
        cell = cells[name]
        time, values, stop = cell["time"], cell["health"], crossings[name]
        time_scale = max(float(time[stop] - time[0]), 1.0)
        for index in range(HISTORY - 1, stop):
            value = float(values[index])
            if boundary is not None:
                if train and value < boundary:
                    continue
                if not train and value >= boundary:
                    continue
            window = values[index - HISTORY + 1:index + 1]
            slopes = []
            for lag in (1, 3, 4):
                start = max(0, index - lag)
                dt = max(float(time[index] - time[start]), 1e-6)
                slopes.append(float((values[index] - values[start]) / dt))
            x.append([
                value,
                value - THRESHOLD,
                *slopes,
                float(window.mean()),
                float(window.std()),
                float((time[index] - time[0]) / time_scale),
                float(min((index + 1) / HISTORY, 1.0)),
            ])
            y.append(float(time[stop] - time[index]))
            groups.append(name)
            health.append(value)
            source.append(cell["source"])
    return {
        "x": np.asarray(x, np.float32),
        "y": np.asarray(y, np.float32),
        "groups": np.asarray(groups),
        "health": np.asarray(health),
        "source": np.asarray(source),
    }


def metric_and_regret(split, prediction, baseline):
    summary = regret_summary(unit_regret(
        split["y"], split["groups"], baseline, prediction
    ))
    return {
        "metrics": regression_metrics(
            split["y"], prediction, split["groups"]
        ),
        "baseline": regression_metrics(
            split["y"], baseline, split["groups"]
        ),
        "regret": {
            "mean": summary[0], "cvar20": summary[1], "maximum": summary[2],
        },
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite LED L92.5 development")

    development = load_file(1)
    stress_cells = load_file(4)
    development_crossings = {
        name: value for name, cell in development.items()
        if (value := crossing(cell)) is not None
    }
    stress_crossings = {
        name: value for name, cell in stress_cells.items()
        if (value := crossing(cell)) is not None
    }
    eligible = sorted(development_crossings)
    permutation = np.random.default_rng(20260910).permutation(eligible)
    first = int(0.6 * len(permutation))
    second = int(0.8 * len(permutation))
    train_ids = list(permutation[:first])
    validation_ids = list(permutation[first:second])
    test_ids = list(permutation[second:])
    stress_ids = sorted(stress_crossings)

    all_train = make_rows(
        development, train_ids, development_crossings
    )
    cutoff = float(np.quantile(all_train["health"], 0.25))
    train = make_rows(
        development, train_ids, development_crossings,
        boundary=cutoff, train=True,
    )
    boundary = float(np.min(train["health"]))
    validation = make_rows(
        development, validation_ids, development_crossings,
        boundary=boundary,
    )
    internal_test = make_rows(
        development, test_ids, development_crossings, boundary=boundary,
    )
    stress_test = make_rows(
        stress_cells, stress_ids, stress_crossings, boundary=boundary,
    )
    combined_test = {
        key: np.concatenate([internal_test[key], stress_test[key]])
        for key in internal_test
    }

    validation_members = {float(trust): [] for trust in TRUSTS}
    test_members = {float(trust): [] for trust in TRUSTS}
    fits = []
    for config in CONFIGS:
        for trust in TRUSTS:
            val, test, epochs = fit_route(
                train, validation, combined_test, config, trust, SEEDS
            )
            validation_members[float(trust)].append(val)
            test_members[float(trust)].append(test)
            fits.append({
                **config, "trust": float(trust), "epochs": epochs,
            })
            print(
                config["width"], trust,
                float(np.sqrt(np.mean(
                    (val.mean(0) - validation["y"]) ** 2
                ))),
                flush=True,
            )
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
    val_prior = np.stack([
        validation_portfolio[float(trust)] for trust in prior_trusts
    ])
    test_prior = np.stack([
        test_portfolio[float(trust)] for trust in prior_trusts
    ])
    decision = fit_auto_regime_prior(
        train["x"], validation["x"], validation["y"], validation["groups"],
        baseline_val, val_prior, prior_trusts,
        n_regimes=2, min_regime_groups=3,
    )
    selected_test = predict_auto_regime_prior(
        decision, combined_test["x"], baseline_test, test_prior, prior_trusts
    )
    support = fit_support_scale(train["x"])
    validation_distance = support_distance(validation["x"], support)
    test_distance = support_distance(combined_test["x"], support)
    support_threshold = float(np.max(validation_distance))
    guarded_test = np.where(
        test_distance <= support_threshold, selected_test, baseline_test
    )

    # Seal model outputs before any test metric is computed.
    np.savez_compressed(
        OUT / "sealed_predictions.npz",
        prediction=selected_test,
        support_guard_prediction=guarded_test,
        baseline=baseline_test,
        groups=combined_test["groups"],
        source=combined_test["source"],
    )
    split = len(internal_test["y"])
    internal_result = metric_and_regret(
        internal_test, selected_test[:split], baseline_test[:split]
    )
    stress_result = metric_and_regret(
        stress_test, selected_test[split:], baseline_test[split:]
    )
    combined_result = metric_and_regret(
        combined_test, selected_test, baseline_test
    )
    guarded_internal = metric_and_regret(
        internal_test, guarded_test[:split], baseline_test[:split]
    )
    guarded_stress = metric_and_regret(
        stress_test, guarded_test[split:], baseline_test[split:]
    )
    guarded_combined = metric_and_regret(
        combined_test, guarded_test, baseline_test
    )
    probability = regime_probabilities(
        decision.regime_map, combined_test["x"]
    )
    payload = {
        "status": "post-screen exploratory external development complete",
        "protocol": "protocols/LED_L925_V14_DEVELOPMENT_PROTOCOL.md",
        "source_bytes": int(sum(
            path.stat().st_size for path in DATA.glob("Data_ID_*.csv")
        )),
        "eligibility": {
            "development": len(development_crossings),
            "stress": len(stress_crossings),
            "split_ids": {
                "train": train_ids, "validation": validation_ids,
                "internal_test": test_ids, "stress_test": stress_ids,
            },
            "rows": {
                "train": len(train["y"]),
                "validation": len(validation["y"]),
                "internal_test": len(internal_test["y"]),
                "stress_test": len(stress_test["y"]),
            },
            "outer_boundary": boundary,
        },
        "decision": {
            "common_scale": decision.common_scale,
            "accepted_regimes": decision.accepted_regimes,
            "regimes": [
                asdict(item) for item in decision.regime_decisions
            ],
            "test_regime_occupancy": [
                int(value) for value in np.bincount(
                    np.argmax(probability, axis=1), minlength=2
                )
            ],
        },
        "internal_test": internal_result,
        "regime_shift_stress": stress_result,
        "combined": combined_result,
        "posthoc_support_guard": {
            "status": "post-test repair ablation; not confirmation",
            "validation_max_distance_threshold": support_threshold,
            "out_of_support_rows": {
                "internal_test": int(np.sum(
                    test_distance[:split] > support_threshold
                )),
                "stress_test": int(np.sum(
                    test_distance[split:] > support_threshold
                )),
            },
            "internal_test": guarded_internal,
            "regime_shift_stress": guarded_stress,
            "combined": guarded_combined,
        },
        "fits": fits,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "decision": payload["decision"],
        "internal": internal_result["metrics"]["pooled"],
        "stress": stress_result["metrics"]["pooled"],
        "combined": combined_result["metrics"]["pooled"],
        "combined_regret": combined_result["regret"],
        "guarded_stress": guarded_stress["metrics"]["pooled"],
        "guarded_combined_regret": guarded_combined["regret"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
