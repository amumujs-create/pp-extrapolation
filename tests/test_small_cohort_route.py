from types import SimpleNamespace

from pp_extrapolation.small_cohort_route import select_ccmr_v22_route


def model(**overrides):
    values = {
        "validation_mean_regret": -0.01,
        "validation_cvar_regret": 0.0,
        "validation_max_regret": 0.0,
        "validation_active_fraction": 0.8,
        "validation_macro_improvement": 0.2,
        "deployment_mass": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def cautious():
    return {
        "pooled_improvement": 0.0,
        "macro_improvement": 0.0,
        "raw_regret": {"mean": 0.0, "cvar20": 0.0, "maximum": 0.0},
    }


def test_small_validation_cohort_uses_crossfit_bank():
    decision = select_ccmr_v22_route(model(), 4, 0.0, cautious())
    assert decision.route == "small_crossfit_bank"
    assert decision.approved


def test_two_validation_units_cannot_use_small_route():
    decision = select_ccmr_v22_route(model(), 2, 0.0, cautious())
    assert decision.route == "exact_fallback"


def test_small_route_rejects_raw_risk_violation():
    decision = select_ccmr_v22_route(
        model(validation_max_regret=0.021), 4, 0.0, cautious()
    )
    assert decision.route == "exact_fallback"


def test_large_validation_cohort_requires_causal_coverage():
    decision = select_ccmr_v22_route(model(), 20, 0.05, cautious())
    assert decision.route == "exact_fallback"
