"""Transferability gate for the unified PP-X executor.

This gate selects a prediction *path* from development evidence. It is not an
uncertainty probability and never accepts test targets.

Versioning
----------
- ``final`` (default, PP-X Final): only ``known_boundary`` decides the prior.
  BQ if True, affine if False. Unused OOF/group/mode checks are not executed.
- ``v1_declared``: archived declared ladder. Kept for historical audits
  (DS03, FEMTO). Those fields were never computed in the Final 9-setting
  portfolio.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math


PRIOR_GATE_VERSION = "final"


@dataclass(frozen=True)
class PriorEvidence:
    known_boundary: bool
    complete_groups: int
    minimum_complete_groups_per_regime: int
    oof_prior_regret: float | None = None
    oof_mode_stability: float | None = None


@dataclass(frozen=True)
class GateDecision:
    route: str
    prior_weight: float
    reason: str
    evidence: dict


def _finite_or_none(evidence: PriorEvidence) -> dict:
    values = asdict(evidence)
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite or None")
    return values


def select_ppx_route(
    evidence: PriorEvidence,
    *,
    min_groups: int = 5,
    min_per_regime: int = 2,
    max_oof_regret: float = 0.0,
    min_mode_stability: float = 0.6,
    version: str | None = None,
) -> GateDecision:
    """PP-X Final prior gate. Unused v1 arguments are accepted and ignored."""
    del min_groups, min_per_regime, max_oof_regret, min_mode_stability
    selected = version or PRIOR_GATE_VERSION
    if selected == "v1_declared":
        return select_ppx_route_v1_declared(evidence)
    if selected != "final":
        raise ValueError(f"unknown prior-gate version: {selected}")
    values = _finite_or_none(evidence)
    if evidence.known_boundary:
        return GateDecision("boundary_pp", 1.0, "declared failure boundary", values)
    # Final: no OOF regret / group-count / mode-stability computation exists,
    # so those checks are not executed. Affine prior is used when the
    # failure boundary is unknown.
    return GateDecision(
        "transferable_prior_pp",
        1.0,
        "final: unknown boundary uses affine prior",
        values,
    )


def select_ppx_route_v1_declared(
    evidence: PriorEvidence,
    *,
    min_groups: int = 5,
    min_per_regime: int = 2,
    max_oof_regret: float = 0.0,
    min_mode_stability: float = 0.6,
) -> GateDecision:
    """Archived v1 declared ladder. Not used by PP-X Final.

    These checks remain only because some historical experiments called them.
    ``oof_mode_stability`` was never computed for the Final 9-setting portfolio.
    """
    values = _finite_or_none(evidence)
    if evidence.known_boundary:
        return GateDecision("boundary_pp", 1.0, "declared failure boundary", values)
    # --- v1_declared unused Final checks (kept for historical audits) ---
    if evidence.complete_groups < min_groups:
        return GateDecision("neural_safety", 0.0, "too few complete source groups", values)
    if evidence.minimum_complete_groups_per_regime < min_per_regime:
        return GateDecision(
            "neural_safety", 0.0, "insufficient complete groups per regime", values
        )
    if evidence.oof_prior_regret is None or evidence.oof_mode_stability is None:
        return GateDecision("neural_safety", 0.0, "transfer evidence unavailable", values)
    if evidence.oof_prior_regret > max_oof_regret:
        return GateDecision(
            "neural_safety", 0.0, "prior loses matched direct model out of fold", values
        )
    if evidence.oof_mode_stability < min_mode_stability:
        return GateDecision("neural_safety", 0.0, "latent mode assignment is unstable", values)
    # --- end v1_declared ---
    return GateDecision(
        "transferable_prior_pp", 1.0, "source OOF transfer evidence approved", values
    )
