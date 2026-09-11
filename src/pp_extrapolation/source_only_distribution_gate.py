"""Source-only global shrinkage gate for corrected predictive distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _numeric_vector(
    value: object, name: str, length: int | None = None
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (length is not None and len(result) != length):
        expected = "" if length is None else f" of length {length}"
        raise ValueError(f"{name} must be a vector{expected}")
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and non-empty")
    return result


def _samples(value: object, name: str, rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != rows or not result.shape[1]:
        raise ValueError(f"{name} must have shape ({rows}, n_samples)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _energy_score(samples: np.ndarray, targets: np.ndarray) -> np.ndarray:
    absolute_error = np.mean(np.abs(samples - targets[:, None]), axis=1)
    pairwise = np.mean(
        np.abs(samples[:, :, None] - samples[:, None, :]), axis=(1, 2)
    )
    return absolute_error - 0.5 * pairwise


def _relative_regret(value: float, fallback: float) -> float:
    if fallback > 0:
        return (value - fallback) / fallback
    return 0.0 if value == 0 else float("inf")


@dataclass(frozen=True)
class UnitDistributionEvidence:
    """One physical unit's cutoff-averaged evidence for one alpha."""

    alpha: float
    unit: object
    cutoff_count: int
    rmse: float
    fallback_rmse: float
    relative_regret: float
    won: bool
    proper_score: float | None
    fallback_proper_score: float | None


@dataclass(frozen=True)
class AlphaDistributionEvidence:
    """Macro and tail-risk criteria evaluated for one global alpha."""

    alpha: float
    unit_count: int
    macro_rmse: float
    fallback_macro_rmse: float
    macro_relative_improvement: float
    win_fraction: float
    worst_relative_regret: float
    cvar20_relative_regret: float
    proper_score_difference: float | None
    passed: bool
    rejection_reasons: tuple[str, ...]
    units: tuple[UnitDistributionEvidence, ...]


@dataclass(frozen=True)
class DistributionGateAudit:
    """Frozen selection thresholds and provenance summary."""

    row_count: int
    source_unit_count: int
    repeated_cutoffs_collapsed: bool
    minimum_units: int
    minimum_macro_improvement: float
    minimum_win_fraction: float
    maximum_worst_relative_regret: float
    maximum_cvar20_relative_regret: float
    proper_score_noninferiority: bool
    proper_score_tolerance: float


@dataclass(frozen=True)
class SourceOnlyDistributionGate:
    """Label-free-at-application gate selected from outer-unit OOF evidence."""

    alpha: float
    source_grade_max: float
    alpha_grid: tuple[float, ...]
    evidence: tuple[AlphaDistributionEvidence, ...]
    audit: DistributionGateAudit
    reason: str

    @property
    def selected_alpha(self) -> float:
        return self.alpha

    @property
    def observed_source_grade_max(self) -> float:
        return self.source_grade_max


def select_source_only_distribution_gate(
    unit_ids: Sequence[object],
    targets: Sequence[float],
    fallback: Sequence[float],
    candidate_mean: Sequence[float],
    *,
    cutoffs: Sequence[object] | None = None,
    source_grades: Sequence[float] | None = None,
    candidate_samples: object | None = None,
    fallback_samples: object | None = None,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    minimum_units: int = 8,
    minimum_macro_improvement: float = 0.02,
    minimum_win_fraction: float = 0.60,
    maximum_worst_relative_regret: float = 0.02,
    maximum_cvar20_relative_regret: float = 0.0,
    proper_score_noninferiority: bool = False,
    proper_score_tolerance: float = 0.0,
) -> SourceOnlyDistributionGate:
    """Select one global alpha from repeated-cutoff outer-unit OOF predictions.

    Each unit-cutoff pair first contributes one RMSE. Repeated cutoffs are then
    averaged within unit, so every physical unit has exactly one vote.
    """

    units = np.asarray(unit_ids, dtype=object)
    if units.ndim != 1 or not len(units):
        raise ValueError("unit_ids must be a non-empty vector")
    n = len(units)
    y = _numeric_vector(targets, "targets", n)
    base = _numeric_vector(fallback, "fallback", n)
    candidate = _numeric_vector(candidate_mean, "candidate_mean", n)
    cutoff_labels = (
        np.zeros(n, dtype=np.int8)
        if cutoffs is None
        else np.asarray(cutoffs, dtype=object)
    )
    if cutoff_labels.shape != (n,):
        raise ValueError(f"cutoffs must be a vector of length {n}")
    grades = (
        np.zeros(n, dtype=np.float64)
        if source_grades is None
        else _numeric_vector(source_grades, "source_grades", n)
    )
    source_grade_max = (
        float("inf") if source_grades is None else float(np.max(grades))
    )
    alphas = tuple(float(value) for value in alpha_grid)
    if (
        not alphas
        or not np.isfinite(alphas).all()
        or any(not 0 <= value <= 1 for value in alphas)
        or len(set(alphas)) != len(alphas)
    ):
        raise ValueError(
            "alpha_grid must contain unique finite values in [0, 1]"
        )
    alphas = tuple(sorted(alphas))
    if 0.0 not in alphas:
        raise ValueError("alpha_grid must include exact fallback alpha=0")
    if minimum_units < 1:
        raise ValueError("minimum_units must be positive")
    fractions = (
        minimum_macro_improvement,
        minimum_win_fraction,
        maximum_worst_relative_regret,
        maximum_cvar20_relative_regret,
        proper_score_tolerance,
    )
    if not np.isfinite(fractions).all() or not 0 <= minimum_win_fraction <= 1:
        raise ValueError("selection thresholds must be finite and valid")

    candidate_draws = (
        None
        if candidate_samples is None
        else _samples(candidate_samples, "candidate_samples", n)
    )
    fallback_draws = (
        None
        if fallback_samples is None
        else _samples(fallback_samples, "fallback_samples", n)
    )
    if proper_score_noninferiority and candidate_draws is None:
        raise ValueError(
            "candidate_samples are required for proper-score evidence"
        )
    if candidate_draws is not None:
        if fallback_draws is None:
            fallback_draws = np.repeat(
                base[:, None], candidate_draws.shape[1], axis=1
            )
        elif fallback_draws.shape[1] != candidate_draws.shape[1]:
            raise ValueError(
                "candidate and fallback samples must have equal sample counts"
            )

    unique_units = tuple(
        sorted(
            set(units.tolist()),
            key=lambda value: (type(value).__name__, repr(value)),
        )
    )
    evidence: list[AlphaDistributionEvidence] = []
    for alpha in alphas:
        blended = base + alpha * (candidate - base)
        blended_scores: np.ndarray | None = None
        fallback_scores: np.ndarray | None = None
        if candidate_draws is not None and fallback_draws is not None:
            blended_draws = fallback_draws + alpha * (
                candidate_draws - fallback_draws
            )
            blended_scores = _energy_score(blended_draws, y)
            fallback_scores = _energy_score(fallback_draws, y)

        unit_rows: list[UnitDistributionEvidence] = []
        for unit in unique_units:
            unit_mask = units == unit
            unit_cutoffs = tuple(
                sorted(
                    set(cutoff_labels[unit_mask].tolist()),
                    key=lambda value: (type(value).__name__, repr(value)),
                )
            )
            rmses: list[float] = []
            fallback_rmses: list[float] = []
            scores: list[float] = []
            base_scores: list[float] = []
            for cutoff in unit_cutoffs:
                mask = unit_mask & (cutoff_labels == cutoff)
                rmses.append(
                    float(np.sqrt(np.mean((blended[mask] - y[mask]) ** 2)))
                )
                fallback_rmses.append(
                    float(np.sqrt(np.mean((base[mask] - y[mask]) ** 2)))
                )
                if blended_scores is not None and fallback_scores is not None:
                    scores.append(float(np.mean(blended_scores[mask])))
                    base_scores.append(float(np.mean(fallback_scores[mask])))
            rmse = float(np.mean(rmses))
            fallback_rmse = float(np.mean(fallback_rmses))
            score = float(np.mean(scores)) if scores else None
            base_score = float(np.mean(base_scores)) if base_scores else None
            unit_rows.append(
                UnitDistributionEvidence(
                    alpha,
                    unit,
                    len(unit_cutoffs),
                    rmse,
                    fallback_rmse,
                    _relative_regret(rmse, fallback_rmse),
                    rmse < fallback_rmse,
                    score,
                    base_score,
                )
            )

        macro_rmse = float(np.mean([row.rmse for row in unit_rows]))
        fallback_macro = float(
            np.mean([row.fallback_rmse for row in unit_rows])
        )
        improvement = (
            (fallback_macro - macro_rmse) / fallback_macro
            if fallback_macro > 0
            else (0.0 if macro_rmse == 0 else -float("inf"))
        )
        regrets = np.asarray([row.relative_regret for row in unit_rows])
        tail_count = max(1, int(np.ceil(0.20 * len(regrets))))
        cvar20 = float(np.mean(np.sort(regrets)[-tail_count:]))
        proper_difference = (
            float(
                np.mean(
                    [
                        row.proper_score - row.fallback_proper_score
                        for row in unit_rows
                        if row.proper_score is not None
                        and row.fallback_proper_score is not None
                    ]
                )
            )
            if blended_scores is not None
            else None
        )
        reasons: list[str] = []
        if len(unit_rows) < minimum_units:
            reasons.append("insufficient_units")
        if improvement + 1e-15 < minimum_macro_improvement:
            reasons.append("macro_rmse_improvement")
        win_fraction = float(np.mean([row.won for row in unit_rows]))
        if win_fraction + 1e-15 < minimum_win_fraction:
            reasons.append("win_fraction")
        worst_regret = float(np.max(regrets))
        if worst_regret > maximum_worst_relative_regret + 1e-15:
            reasons.append("worst_relative_regret")
        if cvar20 > maximum_cvar20_relative_regret + 1e-15:
            reasons.append("cvar20")
        if proper_score_noninferiority and (
            proper_difference is None
            or proper_difference > proper_score_tolerance + 1e-15
        ):
            reasons.append("proper_score")
        evidence.append(
            AlphaDistributionEvidence(
                alpha,
                len(unit_rows),
                macro_rmse,
                fallback_macro,
                improvement,
                win_fraction,
                worst_regret,
                cvar20,
                proper_difference,
                not reasons,
                tuple(reasons),
                tuple(unit_rows),
            )
        )

    passing = [row for row in evidence if row.alpha > 0 and row.passed]
    selected = (
        min(passing, key=lambda row: (row.macro_rmse, row.alpha))
        if passing
        else None
    )
    alpha = 0.0 if selected is None else selected.alpha
    audit = DistributionGateAudit(
        n,
        len(unique_units),
        cutoffs is not None,
        minimum_units,
        minimum_macro_improvement,
        minimum_win_fraction,
        maximum_worst_relative_regret,
        maximum_cvar20_relative_regret,
        proper_score_noninferiority,
        proper_score_tolerance,
    )
    return SourceOnlyDistributionGate(
        alpha,
        source_grade_max,
        alphas,
        tuple(evidence),
        audit,
        (
            "approved_source_oof_alpha"
            if selected is not None
            else "exact_fallback"
        ),
    )


def apply_source_only_distribution_gate(
    gate: SourceOnlyDistributionGate,
    fallback: Sequence[float],
    candidate_mean: Sequence[float],
    support_grades: Sequence[float],
) -> np.ndarray:
    """Apply a frozen gate without labels or application-batch statistics."""

    base_input = np.asarray(fallback)
    if (
        base_input.ndim != 1
        or not len(base_input)
        or not np.isfinite(base_input).all()
    ):
        raise ValueError("fallback must be a finite non-empty vector")
    candidate = _numeric_vector(
        candidate_mean, "candidate_mean", len(base_input)
    )
    grades = _numeric_vector(support_grades, "support_grades", len(base_input))
    if gate.alpha == 0:
        return base_input.copy()
    output = np.asarray(base_input, dtype=np.float64).copy()
    inside = grades <= gate.source_grade_max
    output[inside] = output[inside] + gate.alpha * (
        candidate[inside] - output[inside]
    )
    return output


select_distribution_gate = select_source_only_distribution_gate
apply_distribution_gate = apply_source_only_distribution_gate
fit_source_only_distribution_gate = select_source_only_distribution_gate
apply_source_only_gate = apply_source_only_distribution_gate
DistributionGateEvidence = AlphaDistributionEvidence
DistributionGateDecision = SourceOnlyDistributionGate
