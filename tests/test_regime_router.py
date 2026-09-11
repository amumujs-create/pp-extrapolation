from types import SimpleNamespace

from pp_extrapolation.regime_router import (
    select_ccmr_regime_route,
    select_ccmr_v19_regime_route,
)


def fit(**overrides):
    values = {
        "validation_active_fraction": 0.8,
        "validation_macro_improvement": 0.2,
        "validation_mean_regret": -0.2,
        "validation_cvar_regret": -0.1,
        "validation_max_regret": -0.08,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_routes_uniformly_safe_validation_to_stable_base():
    decision = select_ccmr_regime_route(fit())
    assert decision.route == "stable_base"
    assert decision.stable_validation_certificate


def test_routes_positive_tail_regret_to_cautious_path():
    decision = select_ccmr_regime_route(
        fit(validation_max_regret=0.001)
    )
    assert decision.route == "cautious_causal"
    assert not decision.stable_validation_certificate


def test_routes_weak_average_gain_to_cautious_path():
    decision = select_ccmr_regime_route(
        fit(validation_macro_improvement=0.09)
    )
    assert decision.route == "cautious_causal"


def test_v19_accepts_replicated_validation_within_risk_caps():
    decision = select_ccmr_v19_regime_route(
        fit(
            validation_macro_improvement=0.06,
            validation_mean_regret=-0.02,
            validation_cvar_regret=0.008,
            validation_max_regret=0.018,
        ),
        validation_unit_count=15,
    )
    assert decision.route == "stable_base"


def test_v19_rejects_small_validation_even_when_scores_pass():
    decision = select_ccmr_v19_regime_route(
        fit(
            validation_macro_improvement=0.20,
            validation_mean_regret=-0.20,
            validation_cvar_regret=-0.10,
            validation_max_regret=-0.08,
        ),
        validation_unit_count=9,
    )
    assert decision.route == "cautious_causal"
