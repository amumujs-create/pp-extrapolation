#!/usr/bin/env python3
"""Same-split matched benchmark on the five CCMR v2.2 development domains.

No Alloy/MultiStage loader is imported. Competitor hyperparameters and
portfolio weights are selected from validation data only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_trajectory_development import (  # noqa: E402
    causal_prediction,
    load_development,
    score,
    selected,
    subset,
)
from pp_extrapolation.causal_dynamics_bank import (  # noqa: E402
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.predictor_portfolio import (  # noqa: E402
    fit_predictor_portfolio,
    predict_predictor_portfolio,
)
from pp_extrapolation.small_cohort_route import select_ccmr_v22_route  # noqa: E402

OUT = ROOT / "results/ccmr_v22_matched_development_benchmark"
SEEDS = tuple(range(42, 47))
RNG = np.random.default_rng(20260911)


def features(rows):
    # Figure-matched competitor contract: all methods see the same context
    # columns. CCMR's internal correction features are not exposed.
    return np.asarray(rows["context"], dtype=float)


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    if not np.any(np.abs(values) > 1e-15):
        return 1.0
    if len(values) <= 20:
        masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
        bits = ((masks >> np.arange(len(values), dtype=np.uint64)) & 1)
        signs = 2.0 * bits.astype(float) - 1.0
        return float(np.mean(
            np.abs(signs @ values / len(values)) >= observed - 1e-15
        ))
    signs = RNG.choice((-1.0, 1.0), size=(200_000, len(values)))
    return float((1 + np.sum(
        np.abs(signs @ values / len(values)) >= observed
    )) / 200_001)


def compact(rows, prediction, choose):
    value = score(rows, prediction, choose)
    return {
        "pooled": value["model"]["pooled"],
        "unit_macro_r2": value["model"]["unit_macro_r2"],
        "per_unit": value["model"]["per_unit"],
        "pooled_improvement": value["pooled_improvement"],
        "macro_improvement": value["macro_improvement"],
        "raw_regret": value["raw_regret"],
    }


def fit_ccmr(train, validation, test, intervals):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    bank = fit_causal_dynamics_bank(
        train["correction"], train["context"], train["y"],
        train["groups"], train["context"][:, 0],
        validation_fit["correction"], validation_fit["context"],
        validation_fit["y"], validation_fit["groups"],
        validation_fit["context"][:, 0],
    )
    validation_candidate, _ = predict_causal_dynamics_bank(
        bank, validation["correction"], validation["context"],
        validation["context"][:, 0],
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    route = select_ccmr_v22_route(
        bank,
        len(np.unique(validation["groups"][validation_choose])),
        float(np.mean(validation_causal["active"][validation_choose])),
        score(validation, validation_gated, validation_choose),
    )
    candidate, _ = predict_causal_dynamics_bank(
        bank, test["correction"], test["context"], test["context"][:, 0]
    )
    gated, _ = causal_prediction(candidate, test, test_choose)
    if route.route in ("stable_bank", "small_crossfit_bank"):
        prediction = candidate
    elif route.route == "cautious_causal":
        prediction = gated
    else:
        prediction = test["context"][:, 0].copy()
    return prediction, route.route


def paired_vs_ccmr(ccmr, competitor):
    units = sorted(set(ccmr["per_unit"]) & set(competitor["per_unit"]))
    if set(ccmr["per_unit"]) != set(competitor["per_unit"]):
        raise RuntimeError("unit alignment failure")
    delta = np.asarray([
        competitor["per_unit"][unit]["rmse"] - ccmr["per_unit"][unit]["rmse"]
        for unit in units
    ])
    boot = RNG.choice(delta, size=(50_000, len(delta)), replace=True).mean(1)
    return {
        "n_units": len(units),
        "ccmr_unit_wins": int(np.sum(delta > 0)),
        "mean_absolute_rmse_reduction": float(np.mean(delta)),
        "unit_bootstrap_ci95": np.quantile(boot, (0.025, 0.975)).tolist(),
        "paired_sign_flip_p": exact_sign_flip(delta),
    }


def run_domain(name, parts):
    train, validation, test, intervals = parts
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    ccmr_prediction, route = fit_ccmr(*parts)
    ccmr = compact(test, ccmr_prediction, test_choose)
    portfolio = fit_predictor_portfolio(
        features(train), train["y"], train["groups"],
        features(validation_fit), validation_fit["y"],
        validation_fit["groups"], anchor_index=0, seeds=SEEDS,
        include_engression=True,
    )
    test_x = features(test)
    predictions = {
        model_name: predictor.predict(test_x)
        for model_name, predictor in zip(portfolio.names, portfolio.predictors)
    }
    predictions["validated_portfolio"] = predict_predictor_portfolio(
        portfolio, test_x
    )
    models = {"CCMR_v2.2": ccmr}
    comparisons = {}
    for model_name, prediction in predictions.items():
        result = compact(test, prediction, test_choose)
        models[model_name] = result
        comparisons[model_name] = paired_vs_ccmr(ccmr, result)
    return {
        "route": route,
        "rows": {
            "train": len(train["y"]),
            "validation_tail": int(np.sum(validation_choose)),
            "test_tail": int(np.sum(test_choose)),
        },
        "units": {
            "train": len(np.unique(train["groups"])),
            "validation_tail": len(np.unique(validation["groups"][validation_choose])),
            "test_tail": len(np.unique(test["groups"][test_choose])),
        },
        "portfolio": {
            "names": list(portfolio.names),
            "weights": portfolio.weights.tolist(),
            "unavailable": list(portfolio.unavailable),
        },
        "models": models,
        "paired_ccmr_vs_competitor": comparisons,
    }


def summarize(domains):
    model_names = sorted(next(iter(domains.values()))["models"])
    rows = {}
    for model in model_names:
        r2 = [value["models"][model]["pooled"]["r2"] for value in domains.values()]
        rmse = [value["models"][model]["pooled"]["rmse"] for value in domains.values()]
        base = [value["models"]["persistence"]["pooled"]["rmse"] for value in domains.values()]
        regrets = [
            value["models"][model]["raw_regret"]["maximum"]
            for value in domains.values()
        ]
        rows[model] = {
            "mean_pooled_r2": float(np.mean(r2)),
            "geometric_rmse_ratio_to_persistence": float(np.exp(np.mean(
                np.log(np.maximum(rmse, 1e-12))
                - np.log(np.maximum(base, 1e-12))
            ))),
            "positive_pooled_improvement_domains": int(sum(
                value["models"][model]["pooled_improvement"] > 0
                for value in domains.values()
            )),
            "maximum_raw_regret": float(np.max(regrets)),
        }
    return rows


def main():
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    development = load_development()
    assert set(development) == {
        "Concrete", "LG_M50T", "SIT_LFP", "RADAR_NMC", "Luminosity"
    }
    domains = {}
    for name, parts in development.items():
        print(f"=== {name}", flush=True)
        domains[name] = run_domain(name, parts)
        print({
            model: round(value["pooled"]["r2"], 4)
            for model, value in domains[name]["models"].items()
        }, flush=True)
    payload = {
        "status": "same-split development benchmark complete",
        "owner": "박진서",
        "holdouts_loaded": False,
        "selection": "validation only",
        "seeds": list(SEEDS),
        "domains": domains,
        "summary": summarize(domains),
        "guardrails": [
            "Development-domain retrospective comparison, not prospective confirmation.",
            "All models use identical train/validation/test-tail rows.",
            "Physical-unit paired tests avoid row-level pseudo-replication.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
