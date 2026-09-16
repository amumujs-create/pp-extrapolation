"""Frozen paper-level PP-X route selection.

**Paper main (Final 9):** ``ppx_forward_selector`` first selects the
contract-computable prior/executor family by minimum validation MSE, then
selects seed/fold-local internal configurations by the same validation-only
rule.  The legacy τ eligibility gate is retained for appendix audits only.

**Audit / feasible-set API:** ``select_paper_ppx`` compares candidates vs
``contract.fallback`` only (strict feasible set); use for appendix replay, not
as the sole paper forward rule.

Model fitting and contract-specific feature construction stay outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .transferability_gate import GateDecision, PriorEvidence, select_ppx_route

if TYPE_CHECKING:
    from .ppx_forward_selector import ForwardSelection

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
    unit_gain_ci_low: float = float("-inf")


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
    min_unit_gain_ci_low: float = float("-inf"),
    prior_gate_version: str | None = None,
) -> PaperPPXDecision:
    """Select one PP-X executor without accepting any test outcome.

    Candidate losses and unit statistics must come from identical
    group-disjoint source/validation folds. More complex paths are admitted
    only when they beat the fallback by the frozen margin and satisfy unit-risk
    requirements. Numerical ties select the simpler path.

    Default thresholds ``(2%, 60%, 1.10)`` are the frozen operational point from
    ``protocols/PPX_THRESHOLD_TUNING_OOF_PROTOCOL.md`` (OOF tune once → freeze).
    Callers must not retune them from test labels on a new dataset.
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
    gate: GateDecision = select_ppx_route(
        prior_evidence, version=prior_gate_version
    )
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
            and candidate.unit_gain_ci_low >= min_unit_gain_ci_low
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


def paper_decision_from_forward(
    forward: "ForwardSelection",
    prior_route: str,
    label_to_executor: dict[str, ExecutorName],
    *,
    fallback_executor: ExecutorName = "direct_fallback",
) -> PaperPPXDecision:
    """Map ``ppx_forward_selector.ForwardSelection`` to ``PaperPPXDecision``."""
    from .ppx_forward_selector import ForwardSelection

    if not isinstance(forward, ForwardSelection):
        raise TypeError("forward must be ForwardSelection")
    executor = label_to_executor.get(forward.selected_label, fallback_executor)
    considered: tuple[ExecutorName, ...] = (
        tuple(
            label_to_executor.get(label, fallback_executor)
            for label in forward.gate_passing
        )
        if forward.gate_passing
        else (fallback_executor,)
    )
    return PaperPPXDecision(
        executor,
        prior_route,
        forward.gate_approved,
        forward.selector_rule,
        forward.validation_mse,
        considered,
    )


@dataclass(frozen=True)
class AutoPaperPPXDecision:
    """Paper decision plus the train-inferred contract that produced it."""

    decision: PaperPPXDecision
    contract: PPXContract
    reasons: dict[str, str]
    support_heterogeneity: float
    dropped: tuple[ExecutorName, ...]


def final_priority_executor(
    contract: PPXContract,
    *,
    history: bool,
    dual_scale: bool,
    transport: bool,
) -> ExecutorName:
    """Pick one Final-like executor from detected structure flags.

    Order matches the frozen 9-setting portfolio: transport, dual-scale,
    bounded BQ, history, then unbounded. Validation losses are not used.
    """
    if transport:
        return "regime_transport"
    if dual_scale:
        return "dual_scale"
    if contract.known_boundary:
        return "bounded"
    if history:
        return "history"
    if contract.known_boundary or contract.ordered_progression:
        return "unbounded"
    return contract.fallback


def select_ppx_from_structure(
    train: object,
    *,
    boundary_value: float | None = None,
    time_key: str | None = None,
    regime_key: str | None = None,
    group_key: str | None = None,
    min_history_points: int = 2,
    min_units_for_support: int = 2,
    dual_scale_min_heterogeneity: float = 0.5,
    fallback: Literal["direct_fallback", "persistence_fallback"] = "direct_fallback",
) -> AutoPaperPPXDecision:
    """Detect optional routes from train structure and apply Final priority."""
    from .ppx_contract_inference import (
        detect_optional_executors,
        infer_ppx_contract_from_train_rows,
    )
    from .transferability_gate import select_ppx_route, PriorEvidence

    inferred = infer_ppx_contract_from_train_rows(
        train,  # type: ignore[arg-type]
        boundary_value=boundary_value,
        time_key=time_key,
        regime_key=regime_key,
        group_key=group_key,
        min_history_points=min_history_points,
        min_units_for_support=min_units_for_support,
        dual_scale_min_heterogeneity=dual_scale_min_heterogeneity,
        fallback=fallback,
    )
    detected = detect_optional_executors(
        train,  # type: ignore[arg-type]
        time_key=time_key,
        regime_key=regime_key,
        group_key=group_key,
        min_history_points=min_history_points,
        min_units_for_support=min_units_for_support,
        dual_scale_min_heterogeneity=dual_scale_min_heterogeneity,
    )
    executor = final_priority_executor(
        inferred.contract,
        history=detected.history,
        dual_scale=detected.dual_scale,
        transport=detected.transport,
    )
    gate = select_ppx_route(
        PriorEvidence(inferred.contract.known_boundary, 2, 1)
    )
    allowed = admissible_executors(inferred.contract)
    dropped = tuple(
        name for name in (
            "dual_scale", "regime_transport", "history"
        )
        if name not in allowed
    )
    return AutoPaperPPXDecision(
        decision=PaperPPXDecision(
            executor,
            gate.route,
            executor != inferred.contract.fallback,
            "structure-detected Final priority",
            0.0,
            (executor,),
        ),
        contract=inferred.contract,
        reasons=detected.reasons,
        support_heterogeneity=detected.support_heterogeneity,
        dropped=dropped,
    )


def select_paper_ppx_from_train(
    train: object,
    prior_evidence: PriorEvidence,
    candidates: tuple[PPXCandidateEvidence, ...],
    *,
    boundary_value: float | None = None,
    time_key: str | None = None,
    regime_key: str | None = None,
    group_key: str | None = None,
    min_history_points: int = 2,
    min_units_for_support: int = 2,
    dual_scale_min_heterogeneity: float = 0.5,
    fallback: Literal["direct_fallback", "persistence_fallback"] = "direct_fallback",
    min_relative_improvement: float = 0.02,
    min_unit_win_fraction: float = 0.60,
    max_worst_unit_rmse_ratio: float = 1.10,
    min_unit_gain_ci_low: float = float("-inf"),
    prior_gate_version: str | None = None,
) -> AutoPaperPPXDecision:
    """Infer the contract from train rows, drop illegal candidates, then select.

    Callers do not set ``causal_history`` or the other flags. The train
    structure turns them on or off. Extra fitted executors that the inferred
    contract does not allow are ignored, not scored.
    """
    from .ppx_contract_inference import infer_ppx_contract_from_train_rows

    inferred = infer_ppx_contract_from_train_rows(
        train,  # type: ignore[arg-type]
        boundary_value=boundary_value,
        time_key=time_key,
        regime_key=regime_key,
        group_key=group_key,
        min_history_points=min_history_points,
        min_units_for_support=min_units_for_support,
        dual_scale_min_heterogeneity=dual_scale_min_heterogeneity,
        fallback=fallback,
    )
    allowed = set(admissible_executors(inferred.contract))
    dropped = tuple(
        candidate.executor
        for candidate in candidates
        if candidate.executor not in allowed
    )
    kept = tuple(
        candidate for candidate in candidates if candidate.executor in allowed
    )
    decision = select_paper_ppx(
        inferred.contract,
        prior_evidence,
        kept,
        min_relative_improvement=min_relative_improvement,
        min_unit_win_fraction=min_unit_win_fraction,
        max_worst_unit_rmse_ratio=max_worst_unit_rmse_ratio,
        min_unit_gain_ci_low=min_unit_gain_ci_low,
        prior_gate_version=prior_gate_version,
    )
    return AutoPaperPPXDecision(
        decision=decision,
        contract=inferred.contract,
        reasons=inferred.reasons,
        support_heterogeneity=inferred.support_heterogeneity,
        dropped=dropped,
    )
