#!/usr/bin/env python3
"""Group-OOF continuous portfolio replay for PP-X v1.2 development."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_v11_complete_structure_ablation import prepare_isu
from pp_extrapolation import (
    combine_portfolio,
    continuous_portfolio_policy_config,
    regression_metrics,
    select_continuous_portfolio,
)
from stanford_ppx_v11_safety import prepare as prepare_stanford

SOURCE = ROOT / "results/ppx_v11_complete_structure_ablation"
OUT = ROOT / "results/ppx_v12_continuous_portfolio_development"
TRUSTS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)


def expert_portfolios(name, cohort, predictions, split):
    values = []
    for trust in TRUSTS:
        keys = [
            row["key"] for row in cohort["grid_rows"]
            if np.isclose(row["trust"], trust)
        ]
        if len(keys) != 4:
            raise RuntimeError(f"expected four architectures at trust {trust}")
        values.append(np.mean([
            predictions[f"{name}_{key}_{split}"].mean(0) for key in keys
        ], axis=0))
    return np.asarray(values)


def replay(name, prepared, source, predictions):
    _, validation, test, _, _ = prepared
    cohort = source["cohorts"][name]
    validation_experts = expert_portfolios(
        name, cohort, predictions, "validation"
    )
    test_experts = expert_portfolios(name, cohort, predictions, "test")
    decision = select_continuous_portfolio(
        validation["y"], validation["groups"], validation_experts,
        np.asarray(TRUSTS), **continuous_portfolio_policy_config(),
    )
    weight = np.asarray(decision.weights)
    validation_prediction = combine_portfolio(validation_experts, weight)
    test_prediction = combine_portfolio(test_experts, weight)
    return {
        "decision": asdict(decision),
        "expert_trusts": TRUSTS,
        "expert_validation": [
            regression_metrics(validation["y"], prediction, validation["groups"])
            for prediction in validation_experts
        ],
        "expert_test": [
            regression_metrics(test["y"], prediction, test["groups"])
            for prediction in test_experts
        ],
        "portfolio_validation": regression_metrics(
            validation["y"], validation_prediction, validation["groups"]
        ),
        "portfolio_test": regression_metrics(
            test["y"], test_prediction, test["groups"]
        ),
        "baseline_portfolio_test": regression_metrics(
            test["y"], test_experts[0], test["groups"]
        ),
    }, test_prediction


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite continuous portfolio replay")
    source = json.loads((SOURCE / "results.json").read_text())
    predictions = np.load(SOURCE / "predictions.npz", allow_pickle=False)
    results, arrays = {}, {}
    for name, prepared in (
        ("Stanford", prepare_stanford()),
        ("ISU_250mAh", prepare_isu()),
    ):
        result, prediction = replay(name, prepared, source, predictions)
        results[name] = result
        arrays[name] = prediction
        print(name, json.dumps({
            "decision": result["decision"],
            "portfolio_test": result["portfolio_test"]["pooled"],
            "baseline_test": result["baseline_portfolio_test"]["pooled"],
        }), flush=True)
    payload = {
        "status": "post-test continuous-portfolio development; not confirmation",
        "model": "PP-X v1.2 group-OOF continuous portfolio",
        "policy": continuous_portfolio_policy_config(),
        "cohorts": results,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
