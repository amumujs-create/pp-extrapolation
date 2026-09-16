"""PP-X validation-only selectors.

Paper v4 figure reproduction uses ``select_min_validation_loss`` for
family-level choices and ``select_replicate_validation_min`` for seed/fold
internal configurations.  Both APIs exclude dataset identity and test arrays.

**Validation eligibility screen** τ = (2%, 60%, 1.10) vs a ladder reference marks
which candidates have sufficient validation evidence vs that baseline. This is
the retained v3/legacy audit policy, not the v4 figure selector. It is
*not* a hard ban: labels outside the eligible set may still win via
**family-wide** min validation MSE when the eligible set is empty.

Selector (evidence-prioritized, not reject-then-direct):

.. math::

    \\hat\\ell = \\begin{cases}
      \\arg\\min_{c \\in \\mathcal{E}} \\mathrm{ValMSE}(c)
        & \\mathcal{E} \\neq \\emptyset \\\\
      \\arg\\min_{c \\in \\mathcal{F}} \\mathrm{ValMSE}(c)
        & \\mathcal{E} = \\emptyset
    \\end{cases}

Spec: ``protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md`` § Executor forward.
Reproduce: ``experiments/ppx_logic_route_benchmark_v1.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Iterable, Literal

import numpy as np

ForwardFamily = Literal["bq_pp", "transport", "history_presets"]
SelectionTier = Literal["eligible_min_val", "family_min_val", "dual_portfolio"]
PriorName = Literal["bq", "affine"]

FROZEN_FORWARD_THRESHOLDS = {
    "min_relative_improvement": 0.02,
    "min_unit_win_fraction": 0.60,
    "max_worst_unit_rmse_ratio": 1.10,
}

_LABEL_COMPLEXITY: dict[str, int] = {
    "unbounded": 2,
    "bounded": 3,
    "dual_scale": 4,
    "pp_core": 3,
    "raw_decay0": 2,
    "transport": 4,
    "pp": 2,
    "gated": 4,
    "basic": 1,
    "moments": 2,
    "multiscale": 3,
    "short": 1,
}


@dataclass(frozen=True)
class ForwardSelection:
    """Outcome-blind validation decision for one fitted candidate family."""

    selected_label: str
    gate_passing: tuple[str, ...]
    """Validation-eligible labels (τ vs ladder). Prefer field name ``eligible_labels`` in new code."""
    gate_approved: bool
    """True iff selection used the eligible pool (incl. dual portfolio), not family-wide fallback."""
    selector_rule: str
    validation_mse: float
    selection_tier: SelectionTier = "family_min_val"

    @property
    def eligible_labels(self) -> tuple[str, ...]:
        return self.gate_passing

    @property
    def selected_from_eligible_pool(self) -> bool:
        return self.gate_approved


@dataclass(frozen=True)
class CandidateMetrics:
    """Validation-only metrics against one explicit reference prediction."""

    validation_mse: float
    relative_mse_gain: float
    unit_win_fraction: float
    worst_unit_rmse_ratio: float
    eligible: bool


@dataclass(frozen=True)
class DataDrivenPPXSelection:
    """Blind executor decision made after the contract-fixed PP-X core."""

    selected_executor: str
    selected_label: str
    used_direct_fallback: bool
    reason: str
    executor_eligible: tuple[str, ...]
    executor_metrics: dict[str, CandidateMetrics]
    final_vs_direct: CandidateMetrics | None


@dataclass(frozen=True)
class ValidationMinSelection:
    """Name-blind minimum-risk choice from one validation candidate menu."""

    selected_label: str
    validation_loss: float
    ordered_candidates: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PriorExecutorSelection:
    """Two-level validation decision over admissible prior families."""

    selected_prior: str
    selected_executor: str
    validation_loss: float
    family_winners: tuple[tuple[str, str, float], ...]


def select_min_validation_loss(
    validation_losses: Mapping[str, float],
) -> ValidationMinSelection:
    """Choose the finite minimum validation loss with a deterministic tie-break.

    Candidate labels describe model arms, never datasets.  Test predictions,
    registry scores, and dataset identities are absent from this API.
    """
    if not validation_losses:
        raise ValueError("at least one validation loss is required")
    losses = {str(label): float(loss) for label, loss in validation_losses.items()}
    if not all(np.isfinite(loss) for loss in losses.values()):
        raise ValueError("validation losses must all be finite")
    ordered = tuple(sorted(losses.items(), key=lambda item: (item[1], item[0])))
    return ValidationMinSelection(ordered[0][0], ordered[0][1], ordered)


def select_replicate_validation_min(
    replicate_validation_losses: Sequence[Mapping[str, float]],
) -> tuple[ValidationMinSelection, ...]:
    """Apply the same validation-only selector independently per seed/fold."""
    if not replicate_validation_losses:
        raise ValueError("at least one seed/fold validation menu is required")
    return tuple(
        select_min_validation_loss(losses) for losses in replicate_validation_losses
    )


def select_prior_executor_family(
    validation_losses: Mapping[str, Mapping[str, float]],
) -> PriorExecutorSelection:
    """Choose prior family, then its executor, using validation loss only.

    The caller/contract decides which prior families and executors are
    computable.  This selector deliberately accepts no boundary flag: a known
    boundary may make BQ admissible but can never force BQ to win.  Each
    family's score is its best executor validation loss, after which the best
    family wins with deterministic lexical tie breaks.
    """
    if not validation_losses:
        raise ValueError("at least one prior family is required")
    winners: list[tuple[str, str, float]] = []
    for prior, executor_losses in validation_losses.items():
        name = str(prior)
        if not name:
            raise ValueError("prior family names must be nonempty")
        selected = select_min_validation_loss(executor_losses)
        winners.append((name, selected.selected_label, selected.validation_loss))
    ordered = tuple(sorted(winners, key=lambda row: (row[2], row[0], row[1])))
    prior, executor, loss = ordered[0]
    return PriorExecutorSelection(prior, executor, loss, ordered)


def _val_mse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2))


def _unit_rmse(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    groups = np.asarray(groups)
    return np.asarray(
        [
            float(np.sqrt(np.mean((y[groups == u] - pred[groups == u]) ** 2)))
            for u in np.unique(groups)
        ]
    )


def candidate_metrics(
    y: np.ndarray,
    groups: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    min_relative_improvement: float = FROZEN_FORWARD_THRESHOLDS["min_relative_improvement"],
    min_unit_win_fraction: float = FROZEN_FORWARD_THRESHOLDS["min_unit_win_fraction"],
    max_worst_unit_rmse_ratio: float = FROZEN_FORWARD_THRESHOLDS["max_worst_unit_rmse_ratio"],
) -> CandidateMetrics:
    """Compute all frozen eligibility statistics from validation data only."""
    ref_mse = _val_mse(y, reference)
    mse = _val_mse(y, candidate)
    ref_unit = _unit_rmse(y, reference, groups)
    cand_unit = _unit_rmse(y, candidate, groups)
    gain = (ref_mse - mse) / ref_mse if ref_mse > 0 else float("-inf")
    wins = float(np.mean(cand_unit < ref_unit))
    worst = float(np.max(cand_unit / np.maximum(ref_unit, 1e-12)))
    eligible = (
        mse < ref_mse * (1.0 - min_relative_improvement)
        and wins >= min_unit_win_fraction
        and worst <= max_worst_unit_rmse_ratio
    )
    return CandidateMetrics(mse, gain, wins, worst, eligible)


def ladder_validation_eligible(
    y: np.ndarray,
    groups: np.ndarray,
    candidate: np.ndarray,
    ladder_reference: np.ndarray,
    *,
    min_relative_improvement: float = FROZEN_FORWARD_THRESHOLDS["min_relative_improvement"],
    min_unit_win_fraction: float = FROZEN_FORWARD_THRESHOLDS["min_unit_win_fraction"],
    max_worst_unit_rmse_ratio: float = FROZEN_FORWARD_THRESHOLDS["max_worst_unit_rmse_ratio"],
) -> bool:
    """True if candidate meets frozen τ vs ladder (validation eligibility screen)."""
    return candidate_metrics(
        y,
        groups,
        candidate,
        ladder_reference,
        min_relative_improvement=min_relative_improvement,
        min_unit_win_fraction=min_unit_win_fraction,
        max_worst_unit_rmse_ratio=max_worst_unit_rmse_ratio,
    ).eligible


def select_data_driven_ppx(
    y_val: np.ndarray,
    groups_val: np.ndarray,
    direct_prediction: np.ndarray,
    core_prediction: np.ndarray,
    executor_predictions: dict[str, np.ndarray],
    *,
    core_executor_label: str = "unbounded",
    ladder_reference_prediction: np.ndarray | None = None,
    train_support_heterogeneity: float | None = None,
    dual_scale_min_heterogeneity: float = 0.5,
) -> DataDrivenPPXSelection:
    """Route only executors using validation evidence.

    The typed contract fixes the BQ or affine core before this call.  This
    function deliberately has no BQ-versus-affine selection step: it compares
    executor candidates with an explicit validation-only ladder reference.

    If at least one candidate passes frozen τ, minimum validation MSE within
    that eligible pool wins (except the declared dual-scale portfolio rule).
    If the eligible pool is empty, minimum validation MSE across the executor
    family wins.  This is the paper-forward evidence-prioritized rule; direct
    performance is retained as a diagnostic and never used to switch routes.
    """
    if not executor_predictions:
        raise ValueError("at least one executor prediction is required")
    menu = dict(executor_predictions)
    # The reference and the retained-core arm must be exactly the same vector.
    menu[core_executor_label] = np.asarray(core_prediction, float)
    ladder_reference = (
        np.asarray(core_prediction, float)
        if ladder_reference_prediction is None
        else np.asarray(ladder_reference_prediction, float)
    )
    executor_metrics = {
        executor: candidate_metrics(
            y_val, groups_val, prediction, ladder_reference
        )
        for executor, prediction in menu.items()
    }
    executor_eligible = tuple(
        executor
        for executor, metrics in executor_metrics.items()
        if metrics.eligible
    )
    if executor_eligible:
        if (
            "dual_scale" in executor_eligible
            and train_support_heterogeneity is not None
            and train_support_heterogeneity >= dual_scale_min_heterogeneity
        ):
            selected_executor = "dual_scale"
            reason = "eligible pool; support-heterogeneity dual-scale portfolio"
        else:
            selected_executor = min(
                executor_eligible,
                key=lambda executor: (
                    executor_metrics[executor].validation_mse,
                    _LABEL_COMPLEXITY.get(executor, 99),
                    executor,
                ),
            )
            reason = "eligible pool; minimum validation MSE"
    else:
        selected_executor = min(
            menu,
            key=lambda executor: (
                _val_mse(y_val, menu[executor]),
                _LABEL_COMPLEXITY.get(executor, 99),
                executor,
            ),
        )
        reason = "empty eligible pool; executor-family minimum validation MSE"

    selected_prediction = menu[selected_executor]
    final_metrics = candidate_metrics(
        y_val, groups_val, selected_prediction, direct_prediction
    )
    return DataDrivenPPXSelection(
        selected_executor,
        selected_executor,
        False,
        reason,
        executor_eligible,
        executor_metrics,
        final_metrics,
    )


def ladder_gate_pass(
    y: np.ndarray,
    groups: np.ndarray,
    candidate: np.ndarray,
    ladder_reference: np.ndarray,
    **kwargs: float,
) -> bool:
    """Alias for :func:`ladder_validation_eligible` (legacy name: "gate PASS")."""
    return ladder_validation_eligible(y, groups, candidate, ladder_reference, **kwargs)


def _pick_min_val(labels: Iterable[str], mses: dict[str, float]) -> str:
    return min(labels, key=lambda label: (mses[label], _LABEL_COMPLEXITY.get(label, 99)))


def _eligible_labels(
    menu: Iterable[str],
    val_predictions: dict[str, np.ndarray],
    ladder_reference: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
) -> list[str]:
    return [
        label
        for label in menu
        if ladder_validation_eligible(
            y_val, groups_val, val_predictions[label], ladder_reference
        )
    ]


def _finalize_selection(
    menu: tuple[str, ...],
    mses: dict[str, float],
    eligible: list[str],
    *,
    dual_portfolio: bool = False,
    family_name: str = "family",
) -> ForwardSelection:
    if eligible:
        if dual_portfolio and "dual_scale" in eligible:
            chosen = "dual_scale"
            tier: SelectionTier = "dual_portfolio"
            rule = "eligible pool → dual_scale (support-heterogeneity portfolio)"
        else:
            chosen = _pick_min_val(eligible, mses)
            tier = "eligible_min_val"
            rule = "eligible pool → min validation MSE (complexity tie-break)"
        approved = chosen in eligible
    else:
        chosen = _pick_min_val(menu, mses)
        tier = "family_min_val"
        rule = f"empty eligible set → {family_name}-wide min validation MSE"
        approved = False
    return ForwardSelection(
        chosen,
        tuple(eligible),
        approved,
        rule,
        mses[chosen],
        tier,
    )


def select_forward_bq(
    menu: tuple[str, ...],
    val_predictions: dict[str, np.ndarray],
    ladder_reference: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    *,
    train_support_heterogeneity: float | None = None,
    dual_scale_min_heterogeneity: float = 0.5,
) -> ForwardSelection:
    """BQ battery: eligibility vs ``prior_only`` ladder; evidence-prioritized selector."""
    mses = {label: _val_mse(y_val, val_predictions[label]) for label in menu}
    eligible = _eligible_labels(menu, val_predictions, ladder_reference, y_val, groups_val)
    dual_portfolio = (
        train_support_heterogeneity is not None
        and train_support_heterogeneity >= dual_scale_min_heterogeneity
    )
    return _finalize_selection(
        menu,
        mses,
        eligible,
        dual_portfolio=dual_portfolio,
        family_name="PP",
    )


def select_forward_transport(
    menu: tuple[str, ...],
    val_predictions: dict[str, np.ndarray],
    ladder_reference: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
) -> ForwardSelection:
    """Affine / regime: eligibility vs ``pp_core`` ladder reference."""
    mses = {label: _val_mse(y_val, val_predictions[label]) for label in menu}
    eligible = _eligible_labels(menu, val_predictions, ladder_reference, y_val, groups_val)
    return _finalize_selection(menu, mses, eligible, family_name="transport")


def select_forward_presets(
    presets: tuple[str, ...],
    val_predictions: dict[str, np.ndarray],
    ladder_reference: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
) -> ForwardSelection:
    """History / latent presets: eligibility vs matched ``direct`` on validation."""
    mses = {p: _val_mse(y_val, val_predictions[p]) for p in presets}
    eligible = _eligible_labels(presets, val_predictions, ladder_reference, y_val, groups_val)
    return _finalize_selection(
        tuple(presets),
        mses,
        eligible,
        family_name="preset",
    )
