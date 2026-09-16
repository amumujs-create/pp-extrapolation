import inspect

import numpy as np

from pp_extrapolation.ppx_forward_selector import (
    select_data_driven_ppx,
    select_forward_bq,
    select_forward_transport,
    select_min_validation_loss,
    select_prior_executor_family,
    select_replicate_validation_min,
)


def test_bq_gate_pass_picks_min_val_among_passing():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0, 0, 1, 1])
    prior = np.array([2.0, 4.0, 6.0, 8.0])
    good = np.array([1.0, 2.0, 3.0, 4.0])
    bad = np.array([1.5, 2.5, 3.5, 4.5])
    fs = select_forward_bq(
        ("unbounded", "bounded"),
        {"unbounded": bad, "bounded": good},
        prior,
        y,
        g,
    )
    assert fs.selected_label == "bounded"
    assert "bounded" in fs.gate_passing
    assert fs.gate_approved
    assert fs.selection_tier == "eligible_min_val"


def test_bq_no_pass_falls_back_to_family_val_min():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0, 0, 1, 1])
    prior = y.copy()
    u = np.array([1.05, 2.05, 3.05, 4.05])
    b = np.array([1.02, 2.02, 3.02, 4.02])
    fs = select_forward_bq(
        ("unbounded", "bounded"),
        {"unbounded": u, "bounded": b},
        prior,
        y,
        g,
    )
    assert fs.selected_label == "bounded"
    assert fs.gate_passing == ()
    assert not fs.gate_approved
    assert fs.selection_tier == "family_min_val"
    assert "family-wide" in fs.selector_rule or "empty eligible" in fs.selector_rule


def test_mich_dual_pass_portfolio():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0, 0, 1, 1])
    prior = np.array([1.2, 2.2, 3.2, 4.2])
    dual = np.array([1.0, 2.0, 3.0, 4.0])
    bounded = np.array([0.99, 1.99, 2.99, 3.99])
    fs = select_forward_bq(
        ("bounded", "dual_scale"),
        {"bounded": bounded, "dual_scale": dual},
        prior,
        y,
        g,
        train_support_heterogeneity=0.65,
    )
    assert fs.selected_label == "dual_scale"
    assert fs.selection_tier == "dual_portfolio"


def test_transport_pass_min_val():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0, 0, 1, 1])
    core = np.array([2.0, 4.0, 6.0, 8.0])
    tr = np.array([1.0, 2.0, 3.0, 4.0])
    fs = select_forward_transport(
        ("pp_core", "transport"),
        {"pp_core": core, "transport": tr},
        core,
        y,
        g,
    )
    assert fs.selected_label == "transport"
    assert fs.gate_approved
    assert fs.selection_tier == "eligible_min_val"


def _staged_inputs():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])
    direct = np.array([2.0, 3.0, 4.0, 5.0])
    bq_core = np.array([1.2, 2.2, 3.2, 4.2])
    affine_core = np.array([1.4, 2.4, 3.4, 4.4])
    bounded = np.array([1.0, 2.0, 3.0, 4.0])
    return y, groups, direct, bq_core, affine_core, bounded


def test_data_router_api_has_no_prior_tournament_inputs():
    parameters = inspect.signature(select_data_driven_ppx).parameters
    assert "prior" not in " ".join(parameters)


def test_data_router_selects_executor_from_fixed_core_without_names():
    y, groups, direct, bq_core, affine_core, bounded = _staged_inputs()
    decision = select_data_driven_ppx(
        y,
        groups,
        direct,
        bq_core,
        {"unbounded": bq_core, "bounded": bounded},
    )
    assert decision.selected_executor == "bounded"
    assert not decision.used_direct_fallback


def test_data_router_is_invariant_to_executor_labels():
    y, groups, direct, bq_core, affine_core, bounded = _staged_inputs()
    decision = select_data_driven_ppx(
        y,
        groups,
        direct,
        bq_core,
        {"core": bq_core, "candidate": bounded},
        core_executor_label="core",
    )
    assert decision.selected_executor == "candidate"


def test_data_router_keeps_core_when_optional_executor_not_eligible():
    y, groups, direct, bq_core, _, _ = _staged_inputs()
    almost_same = bq_core + 0.001
    decision = select_data_driven_ppx(
        y,
        groups,
        direct,
        bq_core,
        {"unbounded": bq_core, "bounded": almost_same},
    )
    assert decision.selected_executor == "unbounded"
    assert not decision.used_direct_fallback


def test_data_router_keeps_family_choice_without_direct_route_switch():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])
    direct = y.copy()
    weak = y + 0.1
    decision = select_data_driven_ppx(
        y,
        groups,
        direct,
        weak,
        {"unbounded": weak},
    )
    assert not decision.used_direct_fallback
    assert decision.selected_label == "unbounded"


def test_data_router_reports_direct_diagnostic_without_overriding_executor():
    y = np.zeros(5)
    groups = np.arange(5)
    direct = np.full(5, 10.0)
    core = np.array([10.9, 7.0, 7.0, 7.0, 7.0])
    optional = np.array([11.99, 6.0, 6.0, 6.0, 6.0])
    decision = select_data_driven_ppx(
        y,
        groups,
        direct,
        core,
        {"unbounded": core, "bounded": optional},
    )
    assert decision.executor_eligible == ("bounded",)
    assert not decision.used_direct_fallback
    assert decision.selected_executor == "bounded"
    assert decision.final_vs_direct is not None
    assert not decision.final_vs_direct.eligible


def test_data_router_uses_explicit_ladder_reference():
    y, groups, direct, core, _, bounded = _staged_inputs()
    ladder = np.array([2.0, 4.0, 6.0, 8.0])
    decision = select_data_driven_ppx(
        y,
        groups,
        direct,
        core,
        {"unbounded": core, "bounded": bounded},
        ladder_reference_prediction=ladder,
    )
    assert set(decision.executor_eligible) == {"unbounded", "bounded"}
    assert decision.selected_executor == "bounded"


def test_data_router_dual_portfolio_requires_train_heterogeneity():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])
    ladder = np.array([2.0, 4.0, 6.0, 8.0])
    bounded = y.copy()
    dual = y + 0.01
    decision = select_data_driven_ppx(
        y,
        groups,
        ladder,
        bounded,
        {"bounded": bounded, "dual_scale": dual},
        core_executor_label="bounded",
        ladder_reference_prediction=ladder,
        train_support_heterogeneity=0.6,
    )
    assert decision.selected_executor == "dual_scale"


def test_validation_min_selector_uses_only_losses_and_stable_tie_break():
    decision = select_min_validation_loss({"z_arm": 2.0, "b_arm": 1.0, "a_arm": 1.0})
    assert decision.selected_label == "a_arm"
    assert decision.validation_loss == 1.0


def test_replicate_validation_min_can_select_different_seed_configs():
    decisions = select_replicate_validation_min(
        (
            {"basic": 1.0, "multiscale": 2.0},
            {"basic": 3.0, "multiscale": 1.0},
        )
    )
    assert tuple(d.selected_label for d in decisions) == ("basic", "multiscale")


def test_validation_min_rejects_nonfinite_or_empty_losses():
    for losses in ({}, {"candidate": float("nan")}):
        try:
            select_min_validation_loss(losses)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid validation menu must raise ValueError")


def test_prior_family_selector_does_not_force_bq_when_boundary_makes_it_available():
    decision = select_prior_executor_family(
        {
            "bq": {"bounded": 2.0, "dual_scale": 1.5},
            "affine": {"bounded": 1.2, "dual_scale": 1.0},
        }
    )
    assert decision.selected_prior == "affine"
    assert decision.selected_executor == "dual_scale"
    assert decision.validation_loss == 1.0


def test_prior_family_selector_can_choose_bq_from_same_tournament():
    decision = select_prior_executor_family(
        {
            "bq": {"bounded": 0.8},
            "affine": {"bounded": 1.0},
        }
    )
    assert decision.selected_prior == "bq"
    assert decision.selected_executor == "bounded"
