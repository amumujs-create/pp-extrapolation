#!/usr/bin/env python3
"""Paper-facing structural ablation for deployed CCMR v2.2.

Development domains only. Alloy/MultiStage are not loaded and must not
be used for selection or retuning.

Question answered: which structural pieces are necessary for
(1) safety (false accept / max regret / exact fallback) and
(2) the documented v2.2 performance claim (SIT small-cohort coverage)?
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
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
from pp_extrapolation.small_cohort_route import (  # noqa: E402
    SmallCohortRouteDecision,
    select_ccmr_v22_route,
)

OUT = ROOT / "results/ccmr_v22_paper_structure_ablation"
EXPERTS = (
    "linear_rate",
    "damped_acceleration",
    "monotone_hinge",
    "nonlinear_residual",
)


def select_v20_route(model, units, causal_coverage, cautious_metrics):
    """v2.0 routes: stable / cautious / fallback (no small_crossfit)."""
    decision = select_ccmr_v22_route(
        model, units, causal_coverage, cautious_metrics
    )
    if decision.route == "small_crossfit_bank":
        cautious = bool(
            causal_coverage >= 0.10
            and cautious_metrics["pooled_improvement"] >= 0.005
            and cautious_metrics["macro_improvement"] >= 0.005
            and cautious_metrics["raw_regret"]["mean"] <= 0.0
            and cautious_metrics["raw_regret"]["cvar20"] <= 0.01
            and cautious_metrics["raw_regret"]["maximum"] <= 0.02
        )
        if cautious:
            return SmallCohortRouteDecision(
                "cautious_causal",
                True,
                "v20 ablation: small_crossfit disabled, cautious retained",
            )
        return SmallCohortRouteDecision(
            "exact_fallback",
            False,
            "v20 ablation: small_crossfit disabled, evidence insufficient",
        )
    return decision


def select_cautious_only(model, units, causal_coverage, cautious_metrics):
    del model, units
    cautious = bool(
        causal_coverage >= 0.10
        and cautious_metrics["pooled_improvement"] >= 0.005
        and cautious_metrics["macro_improvement"] >= 0.005
        and cautious_metrics["raw_regret"]["mean"] <= 0.0
        and cautious_metrics["raw_regret"]["cvar20"] <= 0.01
        and cautious_metrics["raw_regret"]["maximum"] <= 0.02
    )
    if cautious:
        return SmallCohortRouteDecision(
            "cautious_causal",
            True,
            "ablation: bank routes disabled, cautious only",
        )
    return SmallCohortRouteDecision(
        "exact_fallback",
        False,
        "ablation: bank routes disabled and cautious failed",
    )


def select_always_fallback(*_args, **_kwargs):
    return SmallCohortRouteDecision(
        "exact_fallback",
        False,
        "ablation: always exact persistence",
    )


def select_always_raw_bank(model, *_args, **_kwargs):
    if model.deployment_mass <= 0:
        return SmallCohortRouteDecision(
            "exact_fallback",
            False,
            "ablation: raw bank requested but mass=0",
        )
    return SmallCohortRouteDecision(
        "stable_bank",
        True,
        "ablation: always deploy certified bank candidate",
    )


ROUTERS = {
    "full_v22": select_ccmr_v22_route,
    "v20_no_small_crossfit": select_v20_route,
    "always_exact_fallback": select_always_fallback,
    "always_raw_bank": select_always_raw_bank,
    "cautious_only": select_cautious_only,
}


def _fit_kwargs(variant: str):
    kwargs = {}
    if variant.startswith("only_"):
        kwargs["expert_names"] = (variant.replace("only_", ""),)
    elif variant.startswith("drop_"):
        drop = variant.replace("drop_", "")
        kwargs["expert_names"] = tuple(name for name in EXPERTS if name != drop)
    if variant == "no_risk_caps":
        kwargs.update(
            validation_mean_cap=1e12,
            validation_cvar_cap=1e12,
            validation_max_cap=1e12,
        )
    if variant in ("relaxed_consensus", "ungated_bank"):
        kwargs.update(sign_threshold=0.50, dispersion_threshold=1e12)
    return kwargs


def _postprocess_model(model, variant: str):
    if variant == "no_residual_bound":
        return replace(model, residual_bound=1e12)
    if variant == "ungated_bank":
        return replace(
            model,
            sign_threshold=0.0,
            dispersion_threshold=1e12,
            support_threshold=1e12,
        )
    if variant == "relaxed_consensus":
        return replace(
            model,
            sign_threshold=0.50,
            dispersion_threshold=1e12,
        )
    return model


def _router_name(variant: str) -> str:
    if variant in ROUTERS:
        return variant
    if variant in (
        "no_risk_caps",
        "no_residual_bound",
        "relaxed_consensus",
        "ungated_bank",
    ) or variant.startswith("only_") or variant.startswith("drop_"):
        return "full_v22"
    raise KeyError(variant)


def deploy(route, test, test_candidate, test_evidence, test_gated, test_causal, choose):
    if route.route in ("stable_bank", "small_crossfit_bank"):
        deployed = test_candidate
        coverage = float(np.mean(test_evidence["active"][choose]))
    elif route.route == "cautious_causal":
        deployed = test_gated
        coverage = float(np.mean(test_causal["active"][choose]))
    else:
        deployed = test["context"][:, 0].copy()
        coverage = 0.0
    metrics = score(test, deployed, choose)
    anchor = test["context"][choose, 0]
    prediction = deployed[choose]
    fallback = prediction == anchor
    fallback_error = (
        float(np.max(np.abs(prediction[fallback] - anchor[fallback])))
        if np.any(fallback)
        else 0.0
    )
    return {
        "route": asdict(route),
        "coverage": coverage,
        "fallback_error": fallback_error,
        "pooled_r2": metrics["model"]["pooled"]["r2"],
        "pooled_rmse": metrics["model"]["pooled"]["rmse"],
        "pooled_improvement": metrics["pooled_improvement"],
        "macro_improvement": metrics["macro_improvement"],
        "raw_regret": metrics["raw_regret"],
        "false_accept": bool(
            route.approved and metrics["pooled_improvement"] < 0
        ),
    }


def run_variant(train, validation, test, intervals, variant: str):
    validation_choose = selected(validation, intervals[1])
    test_choose = selected(test, intervals[2])
    validation_fit = subset(validation, validation_choose)
    model = fit_causal_dynamics_bank(
        train["correction"],
        train["context"],
        train["y"],
        train["groups"],
        train["context"][:, 0],
        validation_fit["correction"],
        validation_fit["context"],
        validation_fit["y"],
        validation_fit["groups"],
        validation_fit["context"][:, 0],
        **_fit_kwargs(variant),
    )
    model = _postprocess_model(model, variant)
    validation_candidate, _ = predict_causal_dynamics_bank(
        model,
        validation["correction"],
        validation["context"],
        validation["context"][:, 0],
    )
    validation_gated, validation_causal = causal_prediction(
        validation_candidate, validation, validation_choose
    )
    cautious_metrics = score(validation, validation_gated, validation_choose)
    units = len(np.unique(validation["groups"][validation_choose]))
    causal_coverage = float(np.mean(validation_causal["active"][validation_choose]))
    router = ROUTERS[_router_name(variant)]
    route = router(model, units, causal_coverage, cautious_metrics)
    test_candidate, test_evidence = predict_causal_dynamics_bank(
        model,
        test["correction"],
        test["context"],
        test["context"][:, 0],
    )
    test_gated, test_causal = causal_prediction(
        test_candidate, test, test_choose
    )
    result = deploy(
        route,
        test,
        test_candidate,
        test_evidence,
        test_gated,
        test_causal,
        test_choose,
    )
    result["validation_units"] = units
    result["validation_causal_coverage"] = causal_coverage
    result["model"] = {
        "selected_candidate": model.selected_candidate,
        "expert_names": [expert.name for expert in model.experts],
        "expert_weights": model.expert_weights.tolist(),
        "deployment_mass": model.deployment_mass,
        "residual_bound": model.residual_bound,
        "validation_macro_improvement": model.validation_macro_improvement,
        "validation_active_fraction": model.validation_active_fraction,
        "validation_raw_regret": {
            "mean": model.validation_mean_regret,
            "cvar20": model.validation_cvar_regret,
            "maximum": model.validation_max_regret,
        },
    }
    return result


VARIANTS = (
    "full_v22",
    "v20_no_small_crossfit",
    "always_exact_fallback",
    "always_raw_bank",
    "cautious_only",
    "no_risk_caps",
    "no_residual_bound",
    "relaxed_consensus",
    "ungated_bank",
    *(f"only_{name}" for name in EXPERTS),
    *(f"drop_{name}" for name in EXPERTS),
)


def summarize(cohorts_by_variant: dict) -> dict:
    summary = {}
    for variant, cohorts in cohorts_by_variant.items():
        improvements = np.asarray([
            value["pooled_improvement"] for value in cohorts.values()
        ], dtype=float)
        regrets = np.asarray([
            value["raw_regret"]["maximum"] for value in cohorts.values()
        ], dtype=float)
        false_accepts = int(sum(value["false_accept"] for value in cohorts.values()))
        sit = cohorts["SIT_LFP"]
        full = cohorts_by_variant["full_v22"]
        sit_loss = (
            sit["pooled_improvement"]
            < full["SIT_LFP"]["pooled_improvement"] - 1e-12
        )
        worse_than_full = int(sum(
            cohorts[name]["pooled_rmse"] > full[name]["pooled_rmse"] + 1e-12
            for name in cohorts
        ))
        summary[variant] = {
            "domains": len(cohorts),
            "false_accepts": false_accepts,
            "maximum_test_raw_regret": float(np.max(regrets)),
            "mean_pooled_improvement": float(np.mean(improvements)),
            "geometric_rmse_ratio_to_full": float(np.exp(np.mean([
                np.log(max(cohorts[name]["pooled_rmse"], 1e-12))
                - np.log(max(full[name]["pooled_rmse"], 1e-12))
                for name in cohorts
            ]))),
            "domains_worse_rmse_than_full": worse_than_full,
            "exact_fallback_all": all(
                value["fallback_error"] == 0 for value in cohorts.values()
            ),
            "sit_pooled_improvement": sit["pooled_improvement"],
            "sit_route": sit["route"]["route"],
            "sit_loses_v22_gain": bool(sit_loss),
            "safety_broken": bool(
                false_accepts > 0
                or float(np.max(regrets)) > 0.02 + 1e-12
                or any(value["fallback_error"] != 0 for value in cohorts.values())
            ),
        }
    return summary


def evidence_claims(summary: dict) -> dict:
    full = summary["full_v22"]
    return {
        "small_crossfit_necessary_for_sit": bool(
            summary["v20_no_small_crossfit"]["sit_loses_v22_gain"]
            and full["sit_pooled_improvement"] > 0
        ),
        "route_necessary_vs_always_raw": bool(
            summary["always_raw_bank"]["safety_broken"]
        ),
        "risk_caps_necessary": bool(
            summary["no_risk_caps"]["safety_broken"]
            or summary["no_risk_caps"]["maximum_test_raw_regret"]
            > full["maximum_test_raw_regret"] + 1e-12
        ),
        "residual_bound_necessary": bool(
            summary["no_residual_bound"]["safety_broken"]
        ),
        "consensus_support_necessary": bool(
            summary["ungated_bank"]["safety_broken"]
            or summary["relaxed_consensus"]["safety_broken"]
        ),
        "single_expert_insufficient": bool(any(
            summary[f"only_{name}"]["safety_broken"]
            or summary[f"only_{name}"]["mean_pooled_improvement"]
            < full["mean_pooled_improvement"] - 1e-12
            for name in EXPERTS
        )),
        "full_v22_safe": not full["safety_broken"],
        "full_v22_sit_positive": full["sit_pooled_improvement"] > 0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    development = load_development()
    # Firewall: never load Alloy/MultiStage in this script.
    assert set(development) == {
        "Concrete", "LG_M50T", "SIT_LFP", "RADAR_NMC", "Luminosity"
    }
    cohorts_by_variant = {}
    for variant in VARIANTS:
        print(f"=== {variant}", flush=True)
        cohorts = {}
        for name, parts in development.items():
            cohorts[name] = run_variant(*parts, variant)
            print(
                f"  {name}: route={cohorts[name]['route']['route']} "
                f"imp={cohorts[name]['pooled_improvement']:.4f} "
                f"regret={cohorts[name]['raw_regret']['maximum']:.4f}",
                flush=True,
            )
        cohorts_by_variant[variant] = cohorts
    summary = summarize(cohorts_by_variant)
    payload = {
        "status": "CCMR v2.2 paper structure ablation complete",
        "owner": "박진서",
        "deployed_model": "CCMR v2.2",
        "holdouts_loaded": False,
        "domains": sorted(development),
        "variants": list(VARIANTS),
        "cohorts": cohorts_by_variant,
        "summary": summary,
        "claims": evidence_claims(summary),
        "protocol": {
            "purpose": (
                "Validate structural necessity and performance contribution "
                "of CCMR v2.2 components for paper evidence"
            ),
            "forbidden": [
                "Alloy_A",
                "MultiStage_RPT",
                "retuning from ablation scores",
            ],
            "metrics": [
                "pooled_improvement",
                "macro_improvement",
                "raw_regret",
                "false_accept",
                "fallback_error",
                "coverage",
                "SIT small-cohort gain retention",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"summary": summary, "claims": payload["claims"]}, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
