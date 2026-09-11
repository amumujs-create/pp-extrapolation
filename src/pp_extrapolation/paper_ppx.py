"""Frozen paper-level PP-X route selection.

This module turns the paper description into one deterministic, test-outcome
blind decision rule. It selects a *fitted prediction path*; model fitting and
contract-specific feature construction remain separate concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .transferability_gate import GateDecision, PriorEvidence, select_ppx_route

ExecutorName = Literal[
    "direct_fallback",
    "persistence_fallback",
    "prior_only",
    "unbounded",
    "bounded",
    "dual_scale",
    "regime_transport",
    "history",
]


@dataclass(frozen=True)
class PPXContract:
    """Outcome-free contract that restricts admissible PP-X executors."""

    known_boundary: bool
    ordered_progression: bool
    causal_history: bool
    observed_regime: bool
    support_heterogeneity_available: bool
    fallback: Literal["direct_fallback", "persistence_fallback"]


@dataclass(frozen=True)
class PPXCandidateEvidence:
    """Group-disjoint validation evidence for one already fitted path."""

    executor: ExecutorName
    validation_loss: float
    unit_win_fraction_vs_fallback: float
    worst_unit_rmse_ratio_vs_fallback: float


@dataclass(frozen=True)
class PaperPPXDecision:
    executor: ExecutorName
    prior_route: str
    approved: bool
    reason: str
    validation_loss: float
    considered: tuple[ExecutorName, ...]


_COMPLEXITY: dict[ExecutorName, int] = {
    "direct_fallback": 0,
    "persistence_fallback": 0,
    "prior_only": 1,
    "unbounded": 2,
    "bounded": 3,
    "regime_transport": 3,
    "history": 3,
    "dual_scale": 4,
}


def admissible_executors(contract: PPXContract) -> tuple[ExecutorName, ...]:
    """Return the candidate set fixed by the outcome-free contract."""
    values: list[ExecutorName] = [contract.fallback]
    if contract.known_boundary or contract.ordered_progression:
        values.extend(("prior_only", "unbounded", "bounded"))
    if contract.support_heterogeneity_available:
        values.append("dual_scale")
    if contract.observed_regime:
        values.append("regime_transport")
    if contract.causal_history:
        values.append("history")
    return tuple(dict.fromkeys(values))


def select_paper_ppx(
    contract: PPXContract,
    prior_evidence: PriorEvidence,
    candidates: tuple[PPXCandidateEvidence, ...],
    *,
    min_relative_improvement: float = 0.02,
    min_unit_win_fraction: float = 0.60,
    max_worst_unit_rmse_ratio: float = 1.10,
) -> PaperPPXDecision:
    """Select one PP-X executor without accepting any test outcome.

    Candidate losses and unit statistics must come from identical
    group-disjoint source/validation folds. More complex paths are admitted
    only when they beat the fallback by the frozen margin and satisfy unit-risk
    requirements. Numerical ties select the simpler path.
    """
    allowed = admissible_executors(contract)
    by_name = {candidate.executor: candidate for candidate in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("candidate executors must be unique")
    unknown = set(by_name) - set(allowed)
    if unknown:
        raise ValueError(f"contract-inadmissible candidates: {sorted(unknown)}")
    if contract.fallback not in by_name:
        raise ValueError("the contract fallback candidate is required")
    for candidate in candidates:
        if candidate.validation_loss <= 0:
            raise ValueError("validation losses must be positive")
        if not 0 <= candidate.unit_win_fraction_vs_fallback <= 1:
            raise ValueError("unit win fractions must lie in [0, 1]")
        if candidate.worst_unit_rmse_ratio_vs_fallback < 0:
            raise ValueError("worst-unit ratios must be nonnegative")

    fallback = by_name[contract.fallback]
    gate: GateDecision = select_ppx_route(prior_evidence)
    if gate.prior_weight == 0:
        return PaperPPXDecision(
            contract.fallback,
            gate.route,
            False,
            f"prior rejected: {gate.reason}",
            fallback.validation_loss,
            (contract.fallback,),
        )

    threshold = fallback.validation_loss * (1.0 - min_relative_improvement)
    feasible = [fallback]
    for name in allowed:
        candidate = by_name.get(name)
        if candidate is None or name == contract.fallback:
            continue
        if (
            candidate.validation_loss < threshold
            and candidate.unit_win_fraction_vs_fallback
            >= min_unit_win_fraction
            and candidate.worst_unit_rmse_ratio_vs_fallback
            <= max_worst_unit_rmse_ratio
        ):
            feasible.append(candidate)
    selected = min(
        feasible,
        key=lambda value: (
            value.validation_loss,
            _COMPLEXITY[value.executor],
            value.executor,
        ),
    )
    approved = selected.executor != contract.fallback
    reason = (
        "validation gain and physical-unit risk criteria passed"
        if approved
        else "no prior-residual executor passed frozen validation criteria"
    )
    return PaperPPXDecision(
        selected.executor,
        gate.route,
        approved,
        reason,
        selected.validation_loss,
        tuple(value.executor for value in feasible),
    )
