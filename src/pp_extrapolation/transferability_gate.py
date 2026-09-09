"""Transferability gate for the unified PP-X executor.

This gate selects a prediction *path* from development evidence. It is not an
uncertainty probability and never accepts test targets. The neural fallback is
part of PP-X so an inadmissible prior need not be forced onto a new domain.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math


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


def select_ppx_route(evidence: PriorEvidence, *, min_groups: int = 5,
                     min_per_regime: int = 2, max_oof_regret: float = 0.0,
                     min_mode_stability: float = 0.6) -> GateDecision:
    """Select boundary, transferable-prior, or neural safety execution.

    `oof_prior_regret` is prior MSE minus matched-direct MSE, computed without
    target-test labels. Missing OOF evidence cannot approve an unknown prior.
    """
    values=asdict(evidence)
    for key,value in values.items():
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError(f'{key} must be finite or None')
    if evidence.known_boundary:
        return GateDecision('boundary_pp',1.0,'declared failure boundary',values)
    if evidence.complete_groups < min_groups:
        return GateDecision('neural_safety',0.0,'too few complete source groups',values)
    if evidence.minimum_complete_groups_per_regime < min_per_regime:
        return GateDecision('neural_safety',0.0,'insufficient complete groups per regime',values)
    if evidence.oof_prior_regret is None or evidence.oof_mode_stability is None:
        return GateDecision('neural_safety',0.0,'transfer evidence unavailable',values)
    if evidence.oof_prior_regret > max_oof_regret:
        return GateDecision('neural_safety',0.0,'prior loses matched direct model out of fold',values)
    if evidence.oof_mode_stability < min_mode_stability:
        return GateDecision('neural_safety',0.0,'latent mode assignment is unstable',values)
    return GateDecision('transferable_prior_pp',1.0,'source OOF transfer evidence approved',values)
