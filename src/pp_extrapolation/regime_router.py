"""Validation-only routing between stable and cautious CCMR deployment."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeRouteDecision:
    route: str
    stable_validation_certificate: bool
    reason: str


def select_ccmr_regime_route(
    fit,
    *,
    minimum_active_fraction=0.50,
    minimum_macro_improvement=0.10,
    minimum_raw_safety_margin=0.05,
):
    """Select a deployment route without inspecting test observations."""
    limits = (
        fit.validation_mean_regret,
        fit.validation_cvar_regret,
        fit.validation_max_regret,
    )
    stable = bool(
        fit.validation_active_fraction >= minimum_active_fraction
        and fit.validation_macro_improvement >= minimum_macro_improvement
        and all(value <= -minimum_raw_safety_margin for value in limits)
    )
    if stable:
        return RegimeRouteDecision(
            route="stable_base",
            stable_validation_certificate=True,
            reason="uniform validation gain supports the base CCMR correction",
        )
    return RegimeRouteDecision(
        route="cautious_causal",
        stable_validation_certificate=False,
        reason="validation evidence is not uniformly strong",
    )


def select_ccmr_v19_regime_route(fit, validation_unit_count):
    """Route v1.9 using replicated validation and deployment risk caps."""
    stable = bool(
        validation_unit_count >= 10
        and fit.validation_active_fraction >= 0.50
        and fit.validation_macro_improvement >= 0.05
        and fit.validation_mean_regret <= 0.0
        and fit.validation_cvar_regret <= 0.01
        and fit.validation_max_regret <= 0.02
    )
    if stable:
        return RegimeRouteDecision(
            route="stable_base",
            stable_validation_certificate=True,
            reason="replicated validation satisfies deployment risk caps",
        )
    return RegimeRouteDecision(
        route="cautious_causal",
        stable_validation_certificate=False,
        reason="replicated validation certificate is incomplete",
    )
