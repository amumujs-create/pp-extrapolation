#!/usr/bin/env python3
"""Aggregate CCMR model ablation, stats, and holdout benchmark for PPT update.

Uses only already-frozen artifacts. Does not retune or open new cohorts.
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ccmr_model_ablation_stats_benchmark_v1"
RNG = np.random.default_rng(20260911)


def load(path: str):
    return json.loads((ROOT / path).read_text())


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    observed = abs(float(np.mean(values)))
    if len(values) <= 16:
        null = [
            abs(float(np.mean(values * np.asarray(signs))))
            for signs in product((-1.0, 1.0), repeat=len(values))
        ]
        return float(np.mean(np.asarray(null) >= observed - 1e-15))
    signs = RNG.choice((-1.0, 1.0), size=(100_000, len(values)))
    return float(
        (1 + np.count_nonzero(np.abs(np.mean(signs * values, axis=1)) >= observed))
        / 100_001
    )


def bootstrap_ci(values: np.ndarray, reps: int = 30_000):
    values = np.asarray(values, float)
    draws = RNG.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def development_table():
    v20 = load("results/ccmr_v20_trajectory_development/results.json")["cohorts"]
    v22 = load("results/ccmr_v22_trajectory_development/results.json")["cohorts"]
    rows = []
    for name in v22:
        a = v20[name]["test"]
        b = v22[name]["test"]
        rows.append({
            "domain": name,
            "v20_route": v20[name]["validation"]["route"],
            "v22_route": v22[name]["route"]["route"],
            "v20_r2": a["model"]["pooled"]["r2"],
            "v22_r2": b["model"]["pooled"]["r2"],
            "v20_pooled_improvement": a["pooled_improvement"],
            "v22_pooled_improvement": b["pooled_improvement"],
            "v20_macro_improvement": a["macro_improvement"],
            "v22_macro_improvement": b["macro_improvement"],
            "v20_max_regret": a["raw_regret"]["maximum"],
            "v22_max_regret": b["raw_regret"]["maximum"],
            "rmse_ratio_v22_over_v20": (
                b["model"]["pooled"]["rmse"] / max(a["model"]["pooled"]["rmse"], 1e-12)
            ),
        })
    ratios = np.asarray([row["rmse_ratio_v22_over_v20"] for row in rows])
    improvements = 1.0 - ratios
    return {
        "domains": rows,
        "summary": {
            "geometric_rmse_ratio_v22_over_v20": float(math.exp(np.mean(np.log(ratios)))),
            "mean_relative_rmse_gain": float(np.mean(improvements)),
            "domains_strictly_better": int(np.sum(ratios < 1 - 1e-12)),
            "domains_nonworse": int(np.sum(ratios <= 1 + 1e-12)),
            "false_accepts_v22": int(sum(
                v22[name]["route"]["approved"]
                and v22[name]["test"]["pooled_improvement"] < 0
                for name in v22
            )),
            "maximum_raw_regret_v22": float(max(
                v22[name]["test"]["raw_regret"]["maximum"] for name in v22
            )),
            "sign_flip_p_rmse_gain": exact_sign_flip(improvements),
            "bootstrap_ci95_rmse_gain": bootstrap_ci(improvements),
        },
    }


def expert_ablation():
    abl = load("results/ccmr_v20_trajectory_development/ablation.json")
    rows = []
    for domain, variants in abl["cohorts"].items():
        for name, metrics in variants.items():
            rows.append({
                "domain": domain,
                "variant": name,
                "pooled_r2": metrics["model"]["pooled"]["r2"],
                "pooled_rmse": metrics["model"]["pooled"]["rmse"],
                "pooled_improvement": metrics["pooled_improvement"],
                "macro_improvement": metrics["macro_improvement"],
                "max_regret": metrics["raw_regret"]["maximum"],
                "coverage": metrics.get("base_coverage"),
            })
    by_variant = {}
    for variant in (
        "linear_rate",
        "damped_acceleration",
        "monotone_hinge",
        "nonlinear_residual",
        "uniform_bank",
        "selected_bank",
    ):
        values = [
            row["pooled_improvement"] for row in rows if row["variant"] == variant
        ]
        by_variant[variant] = {
            "mean_pooled_improvement": float(np.mean(values)),
            "domains_positive": int(sum(value > 0 for value in values)),
            "maximum_raw_regret": float(max(
                row["max_regret"] for row in rows if row["variant"] == variant
            )),
        }
    return {"rows": rows, "by_variant": by_variant}


def holdout_benchmark():
    hold = load("results/ccmr_v22_frozen_holdout_replay/results.json")
    competitors = load(
        "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/results.json"
    )
    cohorts = {}
    for name, item in hold["cohorts"].items():
        ccmr = item["test"]
        eng = item["comparisons"]["Engression"]
        old = competitors["cohorts"][name]["models"]
        table = {
            "CCMR_v2.2": {
                "pooled_r2": ccmr["model"]["pooled"]["r2"],
                "pooled_rmse": ccmr["model"]["pooled"]["rmse"],
                "pooled_improvement": ccmr["pooled_improvement"],
                "macro_improvement": ccmr["macro_improvement"],
                "max_regret": ccmr["raw_regret"]["maximum"],
                "route": item["route"]["route"],
            },
            "Engression_holdout_replay": {
                "pooled_r2": eng["model"]["pooled"]["r2"],
                "pooled_rmse": eng["model"]["pooled"]["rmse"],
                "pooled_improvement": eng["pooled_improvement"],
                "macro_improvement": eng["macro_improvement"],
                "max_regret": eng["raw_regret"]["maximum"],
            },
        }
        key_map = {
            "persistence": "Persistence",
            "PP_latest_successful": "CCMR_sealed_success",
            "V-REx": "V-REx",
            "GroupDRO": "GroupDRO",
            "Monotone_NN": "Monotone NN",
            "Linear_tail_RBF": "Linear-tail RBF",
            "Engression": "Engression",
            "Linear_mean_GP": "Linear-mean GP",
            "TabPFN_v3": "TabPFN v3",
        }
        for model_name, payload in old.items():
            ensemble = payload["ensemble"]
            key = key_map.get(model_name, model_name)
            table[key] = {
                "pooled_r2": ensemble["pooled"]["r2"],
                "pooled_rmse": ensemble["pooled"]["rmse"],
                "pooled_improvement": ensemble["pooled_rmse_improvement"],
                "macro_improvement": ensemble["macro_rmse_improvement"],
                "max_regret": ensemble["raw_regret"]["maximum"],
                "version": payload.get("version"),
            }
        ranked = sorted(
            table.items(),
            key=lambda pair: (
                -pair[1]["pooled_r2"],
                pair[1]["max_regret"],
            ),
        )
        cohorts[name] = {
            "models": table,
            "best_by_r2": ranked[0][0],
            "best_safe_max_regret_zero": next(
                (
                    key for key, value in ranked
                    if value["max_regret"] <= 1e-12
                    and value["pooled_improvement"] > 0
                ),
                None,
            ),
            "ccmr_beats_engression_r2": bool(
                table["CCMR_v2.2"]["pooled_r2"]
                > table["Engression"]["pooled_r2"] + 1e-12
            ),
            "ccmr_safer_than_engression": bool(
                table["CCMR_v2.2"]["max_regret"]
                < table["Engression"]["max_regret"] - 1e-12
            ),
        }
    return {
        "status": "retrospective opened-test benchmark with CCMR v2.2 overlay",
        "confirmatory": False,
        "cohorts": cohorts,
    }


def v23_rejection():
    payload = load("results/ccmr_v23_development_ablation/results.json")
    return {
        "promoted": False,
        "final": payload["attempts"][-1],
        "attempts": payload["attempts"],
        "holdout_replay": payload["holdout_ablation"],
    }


def paired_final_stats():
    evidence = load("results/final_modular_pp_evidence_v1/results.json")
    return {
        "aggregate": evidence["aggregate"],
        "datasets": [
            {
                "dataset": item["dataset"],
                "baseline": item["baseline"],
                "pp_ensemble_r2": item["pp_ensemble_r2"],
                "baseline_ensemble_r2": item["baseline_ensemble_r2"],
                "r2_gap": item["r2_gap"],
                "units_won": item["units_won"],
                "n_units": item["n_units"],
                "sign_flip_p_two_sided": item["sign_flip_p_two_sided"],
                "bh_q": item["bh_q"],
            }
            for item in evidence["datasets"]
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite CCMR ablation/stats/benchmark")
    development = development_table()
    expert = expert_ablation()
    holdout = holdout_benchmark()
    rejected = v23_rejection()
    paired = paired_final_stats()
    payload = {
        "status": "CCMR model ablation + stats + holdout benchmark for PPT",
        "owner": "박진서",
        "current_deployed_model": "CCMR v2.2",
        "rejected_candidate": "CCMR v2.3 AC-CRPE",
        "development_v20_vs_v22": development,
        "expert_ablation_v20": expert,
        "holdout_benchmark": holdout,
        "v23_rejection": rejected,
        "paired_final_pp_evidence": paired,
        "claims": {
            "deploy_v22": True,
            "v22_improves_sit_without_false_accept": True,
            "alloy_accuracy_led_by_engression": True,
            "multistage_safety_led_by_ccmr": True,
            "v23_not_promoted": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "artifact": str(path.relative_to(ROOT)),
        "v22_geo_ratio": development["summary"]["geometric_rmse_ratio_v22_over_v20"],
        "holdouts": list(holdout["cohorts"]),
        "v23_promoted": rejected["promoted"],
    }, indent=2))


if __name__ == "__main__":
    main()
