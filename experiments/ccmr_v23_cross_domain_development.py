#!/usr/bin/env python3
"""Domain-equal LOCO development for CCMR v2.3 AC-CRPE.

The Alloy A and MultiStage loaders are intentionally not imported here.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
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
from pp_extrapolation.small_cohort_route import (  # noqa: E402
    select_ccmr_v22_route,
)
from pp_extrapolation.causal_dynamics_bank import (  # noqa: E402
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.causal_backtest import (  # noqa: E402
    apply_adaptive_causal_backtest_gate,
    apply_causal_backtest_gate,
)
from pp_extrapolation.contracts import (  # noqa: E402
    admissible_experts,
    trajectory_contract,
)
from pp_extrapolation.predictor_portfolio import (  # noqa: E402
    fit_predictor_portfolio,
    predict_predictor_portfolio,
)

OUT = ROOT / "results/ccmr_v23_cross_domain_development_v10"
V22 = ROOT / "results/ccmr_v22_trajectory_development/results.json"
CONDITION_SHIFT = {
    "Concrete": True,
    "LG_M50T": True,
    "SIT_LFP": False,
    "RADAR_NMC": True,
    "Luminosity": True,
}
POLICIES = {
    "anchor_only": None,
    "weak_anchor": {
        "anchor_mass": 0.10,
        "use_prior": False,
    },
    "strict": {
        "anchor_mass": 0.10,
        "use_prior": True,
        "support_quantile": 0.95,
        "sign_threshold": 0.95,
        "dispersion_threshold": 0.35,
        "minimum_validation_gain": 0.01,
    },
    "balanced": {
        "anchor_mass": 0.05,
        "use_prior": True,
        "support_quantile": 0.99,
        "sign_threshold": 0.90,
        "dispersion_threshold": 0.50,
        "minimum_validation_gain": 0.002,
    },
    "wide": {
        "anchor_mass": 0.10,
        "use_prior": True,
        "support_quantile": 0.995,
        "sign_threshold": 0.85,
        "dispersion_threshold": 0.75,
        "minimum_validation_gain": 0.001,
    },
}


def _features(rows):
    return np.column_stack([rows["context"], rows["correction"]])


def _compact_score(rows, prediction, choose):
    result = score(rows, prediction, choose)
    return {
        "pooled_rmse": result["model"]["pooled"]["rmse"],
        "pooled_r2": result["model"]["pooled"]["r2"],
        "macro_improvement": result["macro_improvement"],
        "pooled_improvement": result["pooled_improvement"],
        "raw_regret": result["raw_regret"],
    }


def _causal_gate(rows, candidate, anchor):
    choose = np.ones(len(rows["y"]), dtype=bool)
    deployed, evidence = apply_adaptive_causal_backtest_gate(
        candidate,
        anchor,
        rows["y"],
        rows["groups"],
        rows["origin"],
        rows["target"],
        choose,
        minimum_history=2,
        maximum_history=5,
    )
    return deployed, float(np.mean(evidence["active"]))


def _causal_anchor(rows, prediction):
    return _causal_gate(rows, prediction, rows["context"][:, 0])


def _strict_causal_gate(rows, candidate, anchor):
    choose = np.ones(len(rows["y"]), dtype=bool)
    deployed, evidence = apply_causal_backtest_gate(
        candidate,
        anchor,
        rows["y"],
        rows["groups"],
        rows["origin"],
        rows["target"],
        choose,
        required_wins=5,
    )
    return deployed, float(np.mean(evidence["active"]))


def _legacy_v22(train, validation, test, intervals):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    model = fit_causal_dynamics_bank(
        train["correction"], train["context"], train["y"],
        train["groups"], train["context"][:, 0],
        validation_fit["correction"], validation_fit["context"],
        validation_fit["y"], validation_fit["groups"],
        validation_fit["context"][:, 0],
    )
    validation_candidate, _ = predict_causal_dynamics_bank(
        model, validation["correction"], validation["context"],
        validation["context"][:, 0],
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    causal_coverage = float(np.mean(
        validation_causal["active"][validation_choose]
    ))
    route = select_ccmr_v22_route(
        model,
        len(np.unique(validation["groups"][validation_choose])),
        causal_coverage,
        score(validation, validation_gated, validation_choose),
    )

    def deploy(rows, choose):
        candidate, _ = predict_causal_dynamics_bank(
            model, rows["correction"], rows["context"],
            rows["context"][:, 0],
        )
        if "origin" in rows and "target" in rows:
            gated, _ = causal_prediction(candidate, rows, choose)
        else:
            gated = candidate
        if route.route in ("stable_bank", "small_crossfit_bank"):
            return candidate
        if route.route == "cautious_causal":
            return gated
        return rows["context"][:, 0].copy()

    return {
        "route": route.route,
        "train": deploy(
            train, np.ones(len(train["y"]), dtype=bool)
        ),
        "validation": deploy(validation, validation_choose),
        "test": deploy(test, test_choose),
    }


def _fit_domain(name, train, validation, test, intervals):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    test_fit = subset(test, test_choose)
    test_all = np.ones(len(test_fit["y"]), dtype=bool)
    legacy = _legacy_v22(train, validation, test, intervals)
    legacy_train = legacy["train"]
    legacy_validation = legacy["validation"][validation_choose]
    legacy_test = legacy["test"][test_choose]
    portfolio = fit_predictor_portfolio(
        _features(train),
        train["y"],
        train["groups"],
        _features(validation_fit),
        validation_fit["y"],
        validation_fit["groups"],
        anchor_index=0,
        seeds=(42, 43, 44),
        include_engression=True,
    )
    train_anchor_raw = predict_predictor_portfolio(
        portfolio, _features(train)
    )
    validation_portfolio = predict_predictor_portfolio(
        portfolio, _features(validation_fit)
    )
    test_portfolio = predict_predictor_portfolio(
        portfolio, _features(test_fit)
    )
    contract = trajectory_contract(
        condition_shift=CONDITION_SHIFT[name]
    )
    minimum_anchor_coverage = (
        0.01 if contract.target.value == "observed_trajectory"
        else 0.20
    )
    policy_results = {}
    policy_predictions = {}
    policy_anchors = {}
    for policy_name, policy in POLICIES.items():
        mass = 0.0 if policy is None else policy["anchor_mass"]
        train_anchor = legacy_train + mass * (
            train_anchor_raw - legacy_train
        )
        validation_raw = legacy_validation + mass * (
            validation_portfolio - legacy_validation
        )
        test_raw = legacy_test + mass * (
            test_portfolio - legacy_test
        )
        validation_anchor, validation_anchor_coverage = _causal_gate(
            validation_fit, validation_raw, legacy_validation
        )
        test_anchor, test_anchor_coverage = _causal_gate(
            test_fit, test_raw, legacy_test
        )
        anchor_certified = bool(
            mass == 0.0
            or validation_anchor_coverage >= minimum_anchor_coverage
        )
        if not anchor_certified:
            train_anchor = legacy_train.copy()
            validation_anchor = legacy_validation.copy()
            test_anchor = legacy_test.copy()
            validation_anchor_coverage = 0.0
            test_anchor_coverage = 0.0
        policy_anchors[policy_name] = test_anchor
        if policy is None:
            deployed = test_anchor.copy()
            policy_results[policy_name] = {
                "approved": False,
                "route": "strong_anchor",
                "validation": None,
                "anchor_mass": mass,
                "anchor_certified": anchor_certified,
                "minimum_anchor_coverage": minimum_anchor_coverage,
                "validation_anchor_coverage": validation_anchor_coverage,
                "test_anchor_coverage": test_anchor_coverage,
                "test": _compact_score(test_fit, deployed, test_all),
                "anchor_test": _compact_score(
                    test_fit, test_anchor, test_all
                ),
                "false_accept": False,
                "fallback_error": 0.0,
            }
            policy_predictions[policy_name] = deployed
            continue
        if not policy["use_prior"]:
            deployed = test_anchor.copy()
            policy_results[policy_name] = {
                "approved": False,
                "route": "weak_strong_anchor",
                "validation": None,
                "anchor_mass": mass,
                "anchor_certified": anchor_certified,
                "minimum_anchor_coverage": minimum_anchor_coverage,
                "validation_anchor_coverage": validation_anchor_coverage,
                "test_anchor_coverage": test_anchor_coverage,
                "test": _compact_score(test_fit, deployed, test_all),
                "anchor_test": _compact_score(
                    test_fit, test_anchor, test_all
                ),
                "false_accept": False,
                "fallback_error": 0.0,
            }
            policy_predictions[policy_name] = deployed
            continue
        bank = fit_causal_dynamics_bank(
            train["correction"],
            train["context"],
            train["y"],
            train["groups"],
            train_anchor,
            validation_fit["correction"],
            validation_fit["context"],
            validation_fit["y"],
            validation_fit["groups"],
            validation_anchor,
            support_quantile=policy["support_quantile"],
            sign_threshold=policy["sign_threshold"],
            dispersion_threshold=policy["dispersion_threshold"],
            validation_mean_cap=0.0,
            validation_cvar_cap=0.01,
            validation_max_cap=0.02,
            expert_names=admissible_experts(contract),
        )
        candidate, evidence = predict_causal_dynamics_bank(
            bank,
            test_fit["correction"],
            test_fit["context"],
            test_anchor,
        )
        approved = bool(
            bank.deployment_mass > 0
            and bank.validation_macro_improvement
            >= policy["minimum_validation_gain"]
            and bank.validation_active_fraction >= 0.10
        )
        if approved:
            deployed, prior_causal_coverage = _strict_causal_gate(
                test_fit, candidate, test_anchor
            )
        else:
            deployed = test_anchor.copy()
            prior_causal_coverage = 0.0
        test_score = _compact_score(test_fit, deployed, test_all)
        anchor_score = _compact_score(
            test_fit, test_anchor, test_all
        )
        selected_anchor = test_anchor
        selected_prediction = deployed
        inactive = (
            ~evidence["active"]
            if approved else np.ones(len(test_fit["y"]), dtype=bool)
        )
        fallback_error = float(np.max(np.abs(
            selected_prediction[inactive] - selected_anchor[inactive]
        ))) if np.any(inactive) else 0.0
        policy_results[policy_name] = {
            "approved": approved,
            "route": "contract_prior" if approved else "strong_anchor",
            "anchor_mass": mass,
            "anchor_certified": anchor_certified,
            "minimum_anchor_coverage": minimum_anchor_coverage,
            "validation_anchor_coverage": validation_anchor_coverage,
            "test_anchor_coverage": test_anchor_coverage,
            "test_prior_causal_coverage": prior_causal_coverage,
            "validation": {
                "selected_candidate": bank.selected_candidate,
                "expert_names": [
                    expert.name for expert in bank.experts
                ],
                "deployment_mass": bank.deployment_mass,
                "macro_improvement": bank.validation_macro_improvement,
                "active_fraction": bank.validation_active_fraction,
                "raw_regret": {
                    "mean": bank.validation_mean_regret,
                    "cvar20": bank.validation_cvar_regret,
                    "maximum": bank.validation_max_regret,
                },
            },
            "test": test_score,
            "anchor_test": anchor_score,
            "false_accept": bool(
                approved
                and test_score["pooled_rmse"]
                > anchor_score["pooled_rmse"] + 1e-12
            ),
            "fallback_error": fallback_error,
        }
        policy_predictions[policy_name] = deployed
    return {
        "contract": asdict(contract),
        "legacy_v22_route": legacy["route"],
        "rows": {
            "train": len(train["y"]),
            "validation": int(np.sum(validation_choose)),
            "test": int(np.sum(test_choose)),
        },
        "units": {
            "train": len(np.unique(train["groups"])),
            "validation": len(np.unique(
                validation["groups"][validation_choose]
            )),
            "test": len(np.unique(test["groups"][test_choose])),
        },
        "portfolio": {
            "names": portfolio.names,
            "weights": portfolio.weights.tolist(),
            "validation_objective": portfolio.validation_objective,
            "validation_macro_rmse": portfolio.validation_macro_rmse,
            "validation_worst_rmse": portfolio.validation_worst_rmse,
            "unavailable": portfolio.unavailable,
        },
        "policies": policy_results,
        "_predictions": policy_predictions,
        "_truth": test_fit["y"],
        "_groups": test_fit["groups"],
        "_persistence": test_fit["context"][:, 0],
        "_legacy_v22": legacy_test,
        "_anchors": policy_anchors,
    }


def _policy_summary(domain_results, names, policy):
    values = [domain_results[name]["policies"][policy] for name in names]
    v22 = json.loads(V22.read_text())["cohorts"]
    log_ratios = [
        math.log(
            value["test"]["pooled_rmse"]
            / v22[name]["test"]["model"]["pooled"]["rmse"]
        )
        for name, value in zip(names, values)
    ]
    return {
        "domains": list(names),
        "geometric_rmse_ratio_to_v22": float(
            math.exp(np.mean(log_ratios))
        ),
        "false_accepts": int(sum(
            value["false_accept"] for value in values
        )),
        "maximum_raw_regret": float(max(
            value["test"]["raw_regret"]["maximum"] for value in values
        )),
        "exact_fallback_all": bool(all(
            value["fallback_error"] == 0.0 for value in values
        )),
    }


def _select_policy(domain_results, names):
    summaries = {
        policy: _policy_summary(domain_results, names, policy)
        for policy in POLICIES
    }
    feasible = [
        (value["geometric_rmse_ratio_to_v22"], policy)
        for policy, value in summaries.items()
        if value["false_accepts"] == 0
        and value["maximum_raw_regret"] <= 0.02
        and value["exact_fallback_all"]
    ]
    if not feasible:
        return "anchor_only", summaries
    return min(feasible)[1], summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    prediction_path = OUT / "predictions.npz"
    if path.exists() or prediction_path.exists():
        raise RuntimeError("refusing to overwrite v2.3 development")
    domain_results = {}
    for name, parts in load_development().items():
        print(f"FIT {name}", flush=True)
        domain_results[name] = _fit_domain(name, *parts)
        print(f"DONE {name}", flush=True)
    names = tuple(domain_results)
    nested = {}
    arrays = {}
    for heldout in names:
        development_names = tuple(
            name for name in names if name != heldout
        )
        policy, summaries = _select_policy(
            domain_results, development_names
        )
        result = domain_results[heldout]
        nested[heldout] = {
            "selected_policy": policy,
            "selected_without_domain": heldout,
            "policy": result["policies"][policy],
            "selection_summaries": summaries,
        }
        arrays[f"{heldout}_truth"] = result["_truth"]
        arrays[f"{heldout}_groups"] = result["_groups"]
        arrays[f"{heldout}_persistence"] = result["_persistence"]
        arrays[f"{heldout}_legacy_v22"] = result["_legacy_v22"]
        arrays[f"{heldout}_strong_anchor"] = result[
            "_anchors"
        ][policy]
        arrays[f"{heldout}_prediction"] = result[
            "_predictions"
        ][policy]
    global_policy, global_summaries = _select_policy(
        domain_results, names
    )
    v22 = json.loads(V22.read_text())["cohorts"]
    ratios = [
        nested[name]["policy"]["test"]["pooled_rmse"]
        / v22[name]["test"]["model"]["pooled"]["rmse"]
        for name in names
    ]
    false_accepts = sum(
        nested[name]["policy"]["false_accept"] for name in names
    )
    maximum_regret = max(
        nested[name]["policy"]["test"]["raw_regret"]["maximum"]
        for name in names
    )
    nonworse = sum(ratio <= 1.0 + 1e-12 for ratio in ratios)
    exact = all(
        nested[name]["policy"]["fallback_error"] == 0.0
        for name in names
    )
    promoted = bool(
        math.exp(float(np.mean(np.log(ratios)))) < 1.0
        and nonworse >= 3
        and false_accepts == 0
        and maximum_regret <= 0.02
        and exact
    )
    public_domains = {}
    for name, result in domain_results.items():
        public_domains[name] = {
            key: value for key, value in result.items()
            if not key.startswith("_")
        }
    payload = {
        "status": (
            "promoted" if promoted else "rejected development model"
        ),
        "holdouts_loaded": False,
        "model": "CCMR v2.3 AC-CRPE",
        "policies": POLICIES,
        "domains": public_domains,
        "nested_loco": nested,
        "global_policy_for_frozen_replay": global_policy,
        "global_policy_selection": global_summaries,
        "summary": {
            "domains": len(names),
            "geometric_rmse_ratio_to_v22": float(
                math.exp(np.mean(np.log(ratios)))
            ),
            "geometric_rmse_improvement_over_v22": float(
                1.0 - math.exp(np.mean(np.log(ratios)))
            ),
            "v22_nonworse_domains": nonworse,
            "false_accepts": int(false_accepts),
            "maximum_test_raw_regret": float(maximum_regret),
            "exact_fallback_all": exact,
            "promoted": promoted,
        },
    }
    np.savez_compressed(prediction_path, **arrays)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
