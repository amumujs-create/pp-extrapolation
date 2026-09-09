"""Source-evidence-only executor selection for the paper PP-X model.

The policy deliberately selects one residual executor.  It prevents the final
method from silently becoming an always-on collection of post-hoc modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Executor = Literal["unbounded", "bounded", "dual_scale"]


@dataclass(frozen=True)
class ExecutorEvidence:
    """Validation losses for candidates evaluated on identical source folds."""

    unbounded_loss: float
    bounded_loss: float | None = None
    dual_scale_loss: float | None = None
    support_heterogeneity: float = 0.0


@dataclass(frozen=True)
class ExecutorDecision:
    executor: Executor
    reason: str


def select_residual_executor(
    evidence: ExecutorEvidence, *, min_relative_improvement: float = 0.02,
    dual_scale_min_heterogeneity: float = 0.5,
) -> ExecutorDecision:
    """Choose one executor from validation evidence without test information.

    A more flexible dual-scale residual must beat both the base and bounded
    candidates by the declared margin and requires heterogeneous support.
    """
    if evidence.unbounded_loss <= 0:
        raise ValueError("unbounded_loss must be positive")
    threshold = evidence.unbounded_loss * (1.0 - min_relative_improvement)
    bounded_ok = evidence.bounded_loss is not None and evidence.bounded_loss < threshold
    best_reference = min(evidence.unbounded_loss, evidence.bounded_loss if bounded_ok else evidence.unbounded_loss)
    dual_ok = (
        evidence.dual_scale_loss is not None
        and evidence.support_heterogeneity >= dual_scale_min_heterogeneity
        and evidence.dual_scale_loss < best_reference * (1.0 - min_relative_improvement)
    )
    if dual_ok:
        return ExecutorDecision("dual_scale", "validation improvement with heterogeneous support")
    if bounded_ok:
        return ExecutorDecision("bounded", "validation improvement over unbounded residual")
    return ExecutorDecision("unbounded", "no approved constrained executor")
