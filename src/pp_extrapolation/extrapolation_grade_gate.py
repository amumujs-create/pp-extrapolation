"""Grade-wise generic extrapolation routing from unit-disjoint OOF evidence.

The fitted gate is deliberately label-free at application time.  Targets are
accepted only while constructing source OOF evidence; prediction accepts a
frozen gate, per-observation grades, and candidate predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


OOFPredictor = Callable[[str, np.ndarray, np.ndarray], np.ndarray]
PredictionCallback = Callable[[str, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GradeEpisode:
    """One candidate's evidence on one held-out unit and depth shell."""

    route: str
    unit: object
    lower_grade: float
    upper_grade: float
    sample_count: int
    rmse: float
    regret: float
    outer_train_units: tuple[object, ...]


@dataclass(frozen=True)
class GradeShellEvidence:
    lower_grade: float
    upper_grade: float
    route: str
    safe: bool
    unit_count: int
    mean_unit_rmse: float | None
    worst_unit_regret: float | None
    reason: str


@dataclass(frozen=True)
class ExtrapolationGradeGate:
    """Frozen, deterministic routing policy learned from source OOF labels."""

    fallback_route: str
    routes: tuple[str, ...]
    shell_edges: tuple[float, ...]
    selected_routes: tuple[str, ...]
    evidence: tuple[GradeShellEvidence, ...]
    episodes: tuple[GradeEpisode, ...]
    minimum_units: int
    maximum_worst_unit_regret: float


@dataclass(frozen=True)
class GradeRouteDecision:
    route: str
    shell_index: int | None
    reason: str


def _vector(value: object, name: str, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return result


def fit_extrapolation_grade_gate(
    unit_ids: Sequence[object],
    ordered_coordinate: Sequence[float],
    targets: Sequence[float],
    *,
    candidate_routes: Sequence[str],
    fallback_route: str,
    candidate_predictions: Mapping[str, Sequence[float]] | None = None,
    prediction_callback: OOFPredictor | None = None,
    shell_edges: Sequence[float] = (0.5, 0.65, 0.8, 0.9, 1.0),
    minimum_units: int = 3,
    minimum_samples_per_unit_shell: int = 1,
    maximum_worst_unit_regret: float = 0.0,
) -> ExtrapolationGradeGate:
    """Fit a gate from outer unit-OOF predictions and within-unit depth shells.

    Exactly one prediction source is required.  A callback is invoked once per
    route and outer held-out unit as ``callback(route, train_idx, heldout_idx)``;
    consequently it cannot accidentally receive held-unit rows as training
    indices.  Precomputed arrays are assumed to already be unit-OOF.
    """
    units = np.asarray(unit_ids, dtype=object)
    n = len(units)
    coordinate = _vector(ordered_coordinate, "ordered_coordinate", n)
    y = _vector(targets, "targets", n)
    routes = tuple(str(route) for route in candidate_routes)
    edges = tuple(float(edge) for edge in shell_edges)
    if units.shape != (n,) or n == 0:
        raise ValueError("unit_ids must be a non-empty vector")
    if len(set(routes)) != len(routes) or fallback_route not in routes:
        raise ValueError("candidate_routes must be unique and include fallback_route")
    if len(edges) < 2 or not np.isfinite(edges).all() or any(
        right <= left for left, right in zip(edges, edges[1:])
    ):
        raise ValueError("shell_edges must be finite and strictly increasing")
    if minimum_units < 1 or minimum_samples_per_unit_shell < 1:
        raise ValueError("minimum evidence thresholds must be positive")
    if not np.isfinite(maximum_worst_unit_regret):
        raise ValueError("maximum_worst_unit_regret must be finite")
    if (candidate_predictions is None) == (prediction_callback is None):
        raise ValueError("provide exactly one prediction source")

    unique_units = tuple(dict.fromkeys(units.tolist()))
    predictions: dict[str, np.ndarray] = {}
    if candidate_predictions is not None:
        if set(candidate_predictions) != set(routes):
            raise ValueError("candidate_predictions keys must exactly match routes")
        predictions = {
            route: _vector(candidate_predictions[route], f"prediction[{route}]", n)
            for route in routes
        }
    else:
        predictions = {route: np.empty(n, dtype=float) for route in routes}
        assert prediction_callback is not None
        for held_unit in unique_units:
            heldout = np.flatnonzero(units == held_unit)
            train = np.flatnonzero(units != held_unit)
            if not len(train):
                raise ValueError("at least two physical units are required")
            for route in routes:
                predictions[route][heldout] = _vector(
                    prediction_callback(route, train.copy(), heldout.copy()),
                    f"callback prediction[{route}]",
                    len(heldout),
                )

    episodes: list[GradeEpisode] = []
    for held_unit in unique_units:
        heldout = np.flatnonzero(units == held_unit)
        train_units = tuple(unit for unit in unique_units if unit != held_unit)
        order = np.argsort(coordinate[heldout], kind="stable")
        ranked = heldout[order]
        # Mid-ranks give every observation one deterministic grade in (0, 1).
        grades = (np.arange(len(ranked), dtype=float) + 0.5) / len(ranked)
        for lower, upper in zip(edges, edges[1:]):
            include_upper = upper == edges[-1]
            mask = (grades >= lower) & (
                (grades <= upper) if include_upper else (grades < upper)
            )
            shell_idx = ranked[mask]
            if len(shell_idx) < minimum_samples_per_unit_shell:
                continue
            fallback_rmse = float(
                np.sqrt(np.mean((predictions[fallback_route][shell_idx] - y[shell_idx]) ** 2))
            )
            for route in routes:
                rmse = float(
                    np.sqrt(np.mean((predictions[route][shell_idx] - y[shell_idx]) ** 2))
                )
                episodes.append(
                    GradeEpisode(
                        route,
                        held_unit,
                        lower,
                        upper,
                        len(shell_idx),
                        rmse,
                        rmse - fallback_rmse,
                        train_units,
                    )
                )

    evidence: list[GradeShellEvidence] = []
    selected: list[str] = []
    for lower, upper in zip(edges, edges[1:]):
        safe_candidates: list[tuple[float, str]] = []
        for route in routes:
            rows = [
                episode
                for episode in episodes
                if episode.route == route
                and episode.lower_grade == lower
                and episode.upper_grade == upper
            ]
            count = len(rows)
            mean_rmse = float(np.mean([row.rmse for row in rows])) if rows else None
            worst_regret = max((row.regret for row in rows), default=None)
            enough = count >= minimum_units
            safe = (
                enough
                and worst_regret is not None
                and worst_regret <= maximum_worst_unit_regret
            )
            reason = (
                "approved"
                if safe
                else ("insufficient_unit_evidence" if not enough else "risk_budget_exceeded")
            )
            evidence.append(
                GradeShellEvidence(
                    lower, upper, route, safe, count, mean_rmse, worst_regret, reason
                )
            )
            if safe and mean_rmse is not None:
                safe_candidates.append((mean_rmse, route))
        # The fallback is exact whenever no candidate has sufficient safe evidence.
        selected.append(
            min(safe_candidates, key=lambda item: (item[0], item[1]))[1]
            if safe_candidates
            else fallback_route
        )
    return ExtrapolationGradeGate(
        fallback_route,
        routes,
        edges,
        tuple(selected),
        tuple(evidence),
        tuple(episodes),
        minimum_units,
        float(maximum_worst_unit_regret),
    )


def select_grade_route(
    gate: ExtrapolationGradeGate, validation_grade: float
) -> GradeRouteDecision:
    """Select from frozen evidence using one scalar grade and no target data."""
    grade = float(validation_grade)
    if not np.isfinite(grade):
        raise ValueError("validation_grade must be finite")
    for index, (lower, upper) in enumerate(
        zip(gate.shell_edges, gate.shell_edges[1:])
    ):
        if grade >= lower and (
            grade < upper or (index == len(gate.selected_routes) - 1 and grade <= upper)
        ):
            route = gate.selected_routes[index]
            return GradeRouteDecision(
                route,
                index,
                "safe_oof_shell" if route != gate.fallback_route else "exact_fallback",
            )
    return GradeRouteDecision(gate.fallback_route, None, "grade_outside_evidence")


def predict_with_grade_gate(
    gate: ExtrapolationGradeGate,
    validation_grades: Sequence[float],
    *,
    candidate_predictions: Mapping[str, Sequence[float]] | None = None,
    prediction_callback: PredictionCallback | None = None,
) -> np.ndarray:
    """Apply frozen routes to precomputed predictions or an indexed callback."""
    grades = np.asarray(validation_grades, dtype=float)
    if grades.ndim != 1 or not np.isfinite(grades).all():
        raise ValueError("validation_grades must be a finite vector")
    if (candidate_predictions is None) == (prediction_callback is None):
        raise ValueError("provide exactly one prediction source")
    decisions = [select_grade_route(gate, grade) for grade in grades]
    output = np.empty(len(grades), dtype=float)
    for route in sorted({decision.route for decision in decisions}):
        indices = np.asarray(
            [i for i, decision in enumerate(decisions) if decision.route == route],
            dtype=int,
        )
        if candidate_predictions is not None:
            if route not in candidate_predictions:
                raise ValueError(f"missing predictions for selected route {route!r}")
            values = _vector(
                candidate_predictions[route], f"prediction[{route}]", len(grades)
            )[indices]
        else:
            assert prediction_callback is not None
            values = _vector(
                prediction_callback(route, indices.copy()),
                f"callback prediction[{route}]",
                len(indices),
            )
        output[indices] = values
    return output
