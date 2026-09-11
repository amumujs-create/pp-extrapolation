import inspect

import numpy as np

from pp_extrapolation.source_only_distribution_gate import (
    apply_source_only_distribution_gate,
    select_source_only_distribution_gate,
)


def _oof(candidate_mode="good"):
    units = np.repeat(np.arange(8), 8)
    cutoffs = np.tile(np.repeat([0.5, 0.8], 4), 8)
    targets = np.tile(np.linspace(-1.0, 1.0, 8), 8)
    fallback = targets + 1.0
    candidate = targets.copy()
    if candidate_mode == "unsafe":
        candidate[units == 0] = targets[units == 0] + 4.0
    grades = np.tile(np.linspace(0.1, 1.0, 8), 8)
    return units, cutoffs, targets, fallback, candidate, grades


def test_safe_candidate_selects_best_global_alpha_after_unit_aggregation():
    units, cutoffs, targets, fallback, candidate, grades = _oof()
    gate = select_source_only_distribution_gate(
        units,
        targets,
        fallback,
        candidate,
        cutoffs=cutoffs,
        source_grades=grades,
    )

    assert gate.alpha == 1.0
    assert gate.reason == "approved_source_oof_alpha"
    assert all(row.unit_count == 8 for row in gate.evidence)
    selected = next(row for row in gate.evidence if row.alpha == gate.alpha)
    assert selected.passed
    assert all(unit.cutoff_count == 2 for unit in selected.units)


def test_tail_risk_criteria_reject_candidate_despite_average_gain():
    units, cutoffs, targets, fallback, candidate, grades = _oof("unsafe")
    gate = select_source_only_distribution_gate(
        units,
        targets,
        fallback,
        candidate,
        cutoffs=cutoffs,
        source_grades=grades,
    )

    assert gate.alpha == 0.0
    assert all(not row.passed for row in gate.evidence if row.alpha > 0)


def test_apply_is_label_free_and_alpha_zero_is_exact_fallback_copy():
    parameters = inspect.signature(
        apply_source_only_distribution_gate
    ).parameters
    assert not {"targets", "labels", "y"} & set(parameters)
    units, cutoffs, targets, fallback, candidate, grades = _oof("unsafe")
    gate = select_source_only_distribution_gate(
        units,
        targets,
        fallback,
        candidate,
        cutoffs=cutoffs,
        source_grades=grades,
    )
    values = np.array([-0.0, 1.25, -8.5], dtype=np.float64)
    output = apply_source_only_distribution_gate(
        gate, values, np.array([9.0, 9.0, 9.0]), np.array([0.2, 0.4, 2.0])
    )

    assert output is not values
    assert np.array_equal(output, values)
    assert np.array_equal(output.view(np.uint64), values.view(np.uint64))


def test_rows_beyond_observed_source_grade_use_exact_fallback():
    units, cutoffs, targets, fallback, candidate, grades = _oof()
    gate = select_source_only_distribution_gate(
        units,
        targets,
        fallback,
        candidate,
        cutoffs=cutoffs,
        source_grades=grades,
    )
    base = np.array([10.0, -0.0, 30.0])
    corrected = np.array([5.0, 5.0, 5.0])
    output = apply_source_only_distribution_gate(
        gate, base, corrected, np.array([0.5, 1.01, 2.0])
    )

    assert output[0] == corrected[0]
    assert np.array_equal(output[1:], base[1:])
    assert np.array_equal(output[1:].view(np.uint64), base[1:].view(np.uint64))
