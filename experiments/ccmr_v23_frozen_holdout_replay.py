#!/usr/bin/env python3
"""Frozen replay and structural ablation for promoted CCMR v2.3."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_trajectory_development import (  # noqa: E402
    score,
    selected,
    subset,
)
from ccmr_v23_cross_domain_development import (  # noqa: E402
    POLICIES,
    _causal_gate,
    _features,
    _legacy_v22,
    _strict_causal_gate,
)
from pp_extrapolation.causal_dynamics_bank import (  # noqa: E402
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.contracts import (  # noqa: E402
    admissible_experts,
    trajectory_contract,
)
from pp_extrapolation.predictor_portfolio import (  # noqa: E402
    fit_predictor_portfolio,
    predict_predictor_portfolio,
)

OUT = ROOT / "results/ccmr_v23_frozen_holdout_replay"
MANIFEST = ROOT / "protocols/CCMR_V23_FROZEN_MANIFEST.json"
V22_PREDICTIONS = (
    ROOT / "results/ccmr_v22_frozen_holdout_replay/predictions.npz"
)


def _verify_manifest():
    manifest = json.loads(MANIFEST.read_text())
    if not manifest["promotion_gate_passed"]:
        raise RuntimeError("manifest does not authorize holdout replay")
    for _, (relative, expected) in manifest["artifacts"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"frozen artifact hash mismatch: {relative}")
    return manifest


def _metrics(rows, prediction, choose):
    value = score(rows, prediction, choose)
    return {
        "pooled": value["model"]["pooled"],
        "unit_macro_r2": value["model"]["unit_macro_r2"],
        "pooled_improvement": value["pooled_improvement"],
        "macro_improvement": value["macro_improvement"],
        "raw_regret": value["raw_regret"],
    }


def _bank_variant(
    train,
    validation_fit,
    test,
    train_anchor,
    validation_anchor,
    test_anchor,
    policy,
    expert_names,
    *,
    risk=True,
    relaxed_consensus=False,
):
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
        sign_threshold=(
            0.50 if relaxed_consensus else policy["sign_threshold"]
        ),
        dispersion_threshold=(
            1e12 if relaxed_consensus
            else policy["dispersion_threshold"]
        ),
        validation_mean_cap=0.0 if risk else 1e12,
        validation_cvar_cap=0.01 if risk else 1e12,
        validation_max_cap=0.02 if risk else 1e12,
        expert_names=expert_names,
    )
    candidate, evidence = predict_causal_dynamics_bank(
        bank, test["correction"], test["context"], test_anchor
    )
    approved = bool(
        bank.deployment_mass > 0
        and bank.validation_macro_improvement
        >= policy["minimum_validation_gain"]
        and bank.validation_active_fraction >= 0.10
    )
    if approved:
        prediction, causal_coverage = _strict_causal_gate(
            test, candidate, test_anchor
        )
    else:
        prediction = test_anchor.copy()
        causal_coverage = 0.0
    return prediction, {
        "approved": approved,
        "selected_candidate": bank.selected_candidate,
        "deployment_mass": bank.deployment_mass,
        "validation_macro_improvement": bank.validation_macro_improvement,
        "validation_active_fraction": bank.validation_active_fraction,
        "validation_raw_regret": {
            "mean": bank.validation_mean_regret,
            "cvar20": bank.validation_cvar_regret,
            "maximum": bank.validation_max_regret,
        },
        "test_active": evidence["active"],
        "test_prior_causal_coverage": causal_coverage,
    }


def evaluate(name, values, policy_name, v22):
    train, validation, test, intervals, _, _, comparator_path = values
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    test_fit = subset(test, test_choose)
    legacy = _legacy_v22(train, validation, test, intervals)
    legacy_train = legacy["train"]
    legacy_validation = legacy["validation"][validation_choose]
    legacy_test = legacy["test"][test_choose]
    portfolio = fit_predictor_portfolio(
        _features(train), train["y"], train["groups"],
        _features(validation_fit), validation_fit["y"],
        validation_fit["groups"], anchor_index=0,
        seeds=(42, 43, 44), include_engression=True,
    )
    train_portfolio = predict_predictor_portfolio(
        portfolio, _features(train)
    )
    validation_portfolio = predict_predictor_portfolio(
        portfolio, _features(validation_fit)
    )
    test_portfolio = predict_predictor_portfolio(
        portfolio, _features(test_fit)
    )
    policy = POLICIES[policy_name]
    mass = 0.0 if policy is None else policy["anchor_mass"]
    train_anchor = legacy_train + mass * (
        train_portfolio - legacy_train
    )
    validation_anchor_raw = legacy_validation + mass * (
        validation_portfolio - legacy_validation
    )
    test_anchor_raw = legacy_test + mass * (
        test_portfolio - legacy_test
    )
    validation_anchor, _ = _causal_gate(
        validation_fit, validation_anchor_raw, legacy_validation
    )
    test_anchor, anchor_coverage = _causal_gate(
        test_fit, test_anchor_raw, legacy_test
    )
    anchor_certified = bool(
        mass == 0.0
        or np.mean(validation_anchor != legacy_validation) >= 0.20
    )
    if not anchor_certified:
        train_anchor = legacy_train.copy()
        validation_anchor = legacy_validation.copy()
        test_anchor = legacy_test.copy()
        anchor_coverage = 0.0
    contract = trajectory_contract(condition_shift=True)
    expert_names = admissible_experts(contract)
    if policy is None or not policy["use_prior"]:
        final = test_anchor.copy()
        final_info = {
            "approved": False,
            "route": "strong_anchor",
            "anchor_mass": mass,
            "anchor_certified": anchor_certified,
        }
    else:
        final, final_info = _bank_variant(
            train, validation_fit, test_fit, train_anchor,
            validation_anchor, test_anchor, policy, expert_names,
        )
        final_info["route"] = (
            "contract_prior" if final_info["approved"]
            else "strong_anchor"
        )
        final_info["anchor_certified"] = anchor_certified
    comparators = np.load(comparator_path)
    if not np.allclose(
        comparators["truth"], test_fit["y"]
    ):
        raise RuntimeError(f"{name}: Engression alignment failure")
    if not np.allclose(
        v22[f"{name}_truth"], test_fit["y"]
    ):
        raise RuntimeError(f"{name}: v2.2 alignment failure")
    if not np.allclose(
        v22[f"{name}_prediction"], legacy_test
    ):
        raise RuntimeError(f"{name}: v2.2 reproduction failure")
    variants = {
        "persistence": test_fit["context"][:, 0].copy(),
        "strong_anchor": test_anchor.copy(),
        "CCMR_v22": legacy_test.copy(),
        "Engression": np.asarray(comparators["Engression"]),
        "AC_CRPE_final": final.copy(),
    }
    ablation_info = {"AC_CRPE_final": final_info}
    if policy is not None and policy["use_prior"]:
        variants["no_contract_all_experts"], ablation_info[
            "no_contract_all_experts"
        ] = _bank_variant(
            train, validation_fit, test_fit, train_anchor,
            validation_anchor, test_anchor, policy,
            ("linear_rate", "damped_acceleration", "monotone_hinge",
             "nonlinear_residual"),
        )
        variants["no_risk_caps"], ablation_info[
            "no_risk_caps"
        ] = _bank_variant(
            train, validation_fit, test_fit, train_anchor,
            validation_anchor, test_anchor, policy, expert_names,
            risk=False,
        )
        variants["relaxed_consensus"], ablation_info[
            "relaxed_consensus"
        ] = _bank_variant(
            train, validation_fit, test_fit, train_anchor,
            validation_anchor, test_anchor, policy, expert_names,
            relaxed_consensus=True,
        )
        for expert in expert_names:
            key = f"only_{expert}"
            variants[key], ablation_info[key] = _bank_variant(
                train, validation_fit, test_fit, train_anchor,
                validation_anchor, test_anchor, policy, (expert,),
            )
        if len(expert_names) > 1:
            for expert in expert_names:
                key = f"drop_{expert}"
                kept = tuple(
                    item for item in expert_names if item != expert
                )
                variants[key], ablation_info[key] = _bank_variant(
                    train, validation_fit, test_fit, train_anchor,
                    validation_anchor, test_anchor, policy, kept,
                )
    results = {}
    prediction_arrays = {}
    for key, prediction in variants.items():
        selected_prediction = prediction
        scoring_rows = test_fit
        scoring_rows = {
            **scoring_rows,
            "context": np.column_stack([
                test_fit["context"][:, 0],
                np.zeros((len(test_fit["y"]), 6)),
            ]),
        }
        results[key] = _metrics(
            scoring_rows,
            selected_prediction,
            np.ones(len(selected_prediction), dtype=bool),
        )
        if key in ablation_info:
            results[key]["fit"] = ablation_info[key]
        prediction_arrays[key] = selected_prediction
    final_selected = final
    selected_anchor = test_anchor
    inactive = final_selected == selected_anchor
    fallback_error = (
        float(np.max(np.abs(
            final_selected[inactive] - selected_anchor[inactive]
        ))) if np.any(inactive) else 0.0
    )
    final_rmse = results["AC_CRPE_final"]["pooled"]["rmse"]
    engression_rmse = results["Engression"]["pooled"]["rmse"]
    return {
        "portfolio": {
            "names": portfolio.names,
            "weights": portfolio.weights.tolist(),
            "test_causal_coverage": anchor_coverage,
            "unavailable": portfolio.unavailable,
        },
        "final_route": final_info,
        "fallback_error": fallback_error,
        "beats_engression": bool(final_rmse < engression_rmse - 1e-12),
        "ties_engression": bool(
            abs(final_rmse - engression_rmse) <= 1e-12
        ),
        "variants": results,
    }, prediction_arrays, test_fit["y"], test_fit["groups"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite v2.3 frozen replay")
    manifest = _verify_manifest()
    # Import only after the frozen source hashes have been verified.
    from ccmr_v20_frozen_holdout_replay import load_holdouts

    v22 = np.load(V22_PREDICTIONS)
    payload = {
        "status": "frozen retrospective holdout replay",
        "confirmatory": False,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "global_policy": manifest["global_policy"],
        "cohorts": {},
    }
    arrays = {}
    for name, values in load_holdouts().items():
        result, predictions, truth, groups = evaluate(
            name, values, manifest["global_policy"], v22
        )
        payload["cohorts"][name] = result
        arrays[f"{name}_truth"] = truth
        arrays[f"{name}_groups"] = groups.astype(str)
        for key, prediction in predictions.items():
            arrays[f"{name}_{key}"] = prediction
        print(name, flush=True)
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
