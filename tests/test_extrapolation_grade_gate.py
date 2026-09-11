import numpy as np
import pytest

from pp_extrapolation.extrapolation_grade_gate import (
    fit_extrapolation_grade_gate,
    predict_with_grade_gate,
    select_grade_route,
)


def _synthetic():
    units = np.repeat(np.arange(4), 20)
    coordinate = np.tile(np.arange(20), 4)
    targets = np.tile(np.linspace(0.0, 1.0, 20), 4)
    grades = np.tile((np.arange(20) + 0.5) / 20, 4)
    fallback = targets + 0.4
    shallow = np.where(grades < 0.75, targets, targets + 0.8)
    deep = np.where(grades >= 0.75, targets, targets + 0.8)
    return units, coordinate, targets, {
        "fallback": fallback,
        "shallow": shallow,
        "deep": deep,
    }


def test_selects_different_safe_routes_by_extrapolation_depth():
    units, coordinate, targets, predictions = _synthetic()
    gate = fit_extrapolation_grade_gate(
        units,
        coordinate,
        targets,
        candidate_routes=("fallback", "shallow", "deep"),
        fallback_route="fallback",
        candidate_predictions=predictions,
        shell_edges=(0.5, 0.75, 1.0),
        minimum_units=4,
    )

    assert select_grade_route(gate, 0.6).route == "shallow"
    assert select_grade_route(gate, 0.9).route == "deep"
    assert select_grade_route(gate, 0.2).route == "fallback"


def test_outer_oof_callback_never_receives_held_unit_in_training():
    units, coordinate, targets, expected = _synthetic()
    calls = []

    def predict(route, train_indices, heldout_indices):
        train_units = set(units[train_indices])
        heldout_units = set(units[heldout_indices])
        assert train_units.isdisjoint(heldout_units)
        assert len(heldout_units) == 1
        calls.append((route, next(iter(heldout_units))))
        return expected[route][heldout_indices]

    gate = fit_extrapolation_grade_gate(
        units,
        coordinate,
        targets,
        candidate_routes=("fallback", "shallow", "deep"),
        fallback_route="fallback",
        prediction_callback=predict,
        shell_edges=(0.5, 0.75, 1.0),
        minimum_units=4,
    )

    assert len(calls) == 12
    assert all(episode.unit not in episode.outer_train_units for episode in gate.episodes)


def test_insufficient_evidence_is_exact_fallback_and_label_free_at_apply():
    units, coordinate, targets, predictions = _synthetic()
    gate = fit_extrapolation_grade_gate(
        units,
        coordinate,
        targets,
        candidate_routes=("fallback", "shallow", "deep"),
        fallback_route="fallback",
        candidate_predictions=predictions,
        shell_edges=(0.5, 0.75, 1.0),
        minimum_units=5,
    )
    fallback = np.array([1.25, -3.5, 8.0], dtype=np.float64)
    output = predict_with_grade_gate(
        gate,
        np.array([0.55, 0.8, 1.2]),
        candidate_predictions={"fallback": fallback},
    )

    assert np.array_equal(output, fallback)
    assert all(route == "fallback" for route in gate.selected_routes)


def test_prediction_callback_is_indexed_and_input_validation_is_strict():
    units, coordinate, targets, predictions = _synthetic()
    gate = fit_extrapolation_grade_gate(
        units,
        coordinate,
        targets,
        candidate_routes=("fallback", "shallow", "deep"),
        fallback_route="fallback",
        candidate_predictions=predictions,
        shell_edges=(0.5, 0.75, 1.0),
        minimum_units=4,
    )
    route_values = {
        "fallback": np.array([10.0, 11.0, 12.0]),
        "shallow": np.array([20.0, 21.0, 22.0]),
        "deep": np.array([30.0, 31.0, 32.0]),
    }

    def callback(route, indices):
        return route_values[route][indices]

    result = predict_with_grade_gate(
        gate, [0.6, 0.9, 0.1], prediction_callback=callback
    )
    assert result.tolist() == [20.0, 31.0, 12.0]

    with pytest.raises(ValueError, match="exactly one"):
        fit_extrapolation_grade_gate(
            units,
            coordinate,
            targets,
            candidate_routes=("fallback",),
            fallback_route="fallback",
        )
