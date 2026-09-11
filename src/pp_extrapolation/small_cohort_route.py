"""Frozen deployment route for CCMR v2.2 small validation cohorts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmallCohortRouteDecision:
    route: str
    approved: bool
    reason: str


def select_ccmr_v22_route(
    model,
    validation_unit_count,
    causal_shadow_coverage,
    cautious_metrics,
):
    """Select stable, small-crossfit, cautious, or exact route."""
    risk_ok = bool(
        model.validation_mean_regret <= 0.0
        and model.validation_cvar_regret <= 0.01
        and model.validation_max_regret <= 0.02
    )
    stable = bool(
        validation_unit_count >= 10
        and model.validation_active_fraction >= 0.50
        and model.validation_macro_improvement >= 0.05
        and causal_shadow_coverage >= 0.10
        and risk_ok
        and model.deployment_mass > 0
    )
    if stable:
        return SmallCohortRouteDecision(
            "stable_bank",
            True,
            "replicated validation and causal-shadow certificate passed",
        )
    small = bool(
        3 <= validation_unit_count < 10
        and model.validation_active_fraction >= 0.50
        and model.validation_macro_improvement >= 0.10
        and risk_ok
        and model.deployment_mass > 0
    )
    if small:
        return SmallCohortRouteDecision(
            "small_crossfit_bank",
            True,
            "small-cohort cross-fit and raw-risk certificate passed",
        )
    cautious = bool(
        causal_shadow_coverage >= 0.10
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
            "causal shadow forecasts satisfy deployment risk caps",
        )
    return SmallCohortRouteDecision(
        "exact_fallback",
        False,
        "validation evidence is insufficient for correction",
    )
