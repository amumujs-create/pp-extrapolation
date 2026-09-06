"""Label-free applicability certificates for typed extrapolation routes."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RegimeCertificate:
    accepted: bool
    train_levels: tuple[str, ...]
    source_levels: tuple[str, ...]
    unseen_levels: tuple[str, ...]
    covered_fraction: float
    reason: str

@dataclass(frozen=True)
class ExtrapolationCertificate:
    """Validation-only decision for selective extrapolation."""
    accepted: bool
    validation_r2: float
    baseline_relative_mse_gain: float
    normalized_seed_disagreement: float
    checks: dict[str, bool]
    reasons: tuple[str, ...]

def certify_extrapolation(
    *, validation_r2: float, baseline_relative_mse_gain: float,
    normalized_seed_disagreement: float, regime_covered: bool,
    minimum_validation_r2: float = 0.0, minimum_relative_gain: float = 0.02,
    maximum_seed_disagreement: float = 0.25,
) -> ExtrapolationCertificate:
    """Accept only when coverage, pseudo-tail skill, gain, and stability pass."""
    values=np.asarray([validation_r2,baseline_relative_mse_gain,
                       normalized_seed_disagreement,minimum_validation_r2,
                       minimum_relative_gain,maximum_seed_disagreement],dtype=float)
    if not np.isfinite(values).all() or minimum_relative_gain<0 or maximum_seed_disagreement<0:
        raise ValueError("certificate inputs and thresholds must be finite and valid")
    checks={
        "regime_covered":bool(regime_covered),
        "positive_pseudo_tail_skill":float(validation_r2)>=float(minimum_validation_r2),
        "beats_baseline_margin":float(baseline_relative_mse_gain)>=float(minimum_relative_gain),
        "seed_stable":float(normalized_seed_disagreement)<=float(maximum_seed_disagreement),
    }
    reasons=tuple(name for name,passed in checks.items() if not passed)
    return ExtrapolationCertificate(not reasons,float(validation_r2),
        float(baseline_relative_mse_gain),float(normalized_seed_disagreement),checks,reasons)

def certify_categorical_regime(train_context, source_context) -> RegimeCertificate:
    """Reject a route when its source contains a regime absent from training."""
    train=np.asarray(train_context).astype(str);source=np.asarray(source_context).astype(str)
    if train.ndim!=1 or source.ndim!=1 or not len(train) or not len(source):
        raise ValueError("train_context and source_context must be nonempty vectors")
    known=set(train);unseen=tuple(sorted(set(source)-known))
    covered=float(np.mean(np.isin(source,list(known))))
    accepted=not unseen
    return RegimeCertificate(accepted,tuple(sorted(known)),tuple(sorted(set(source))),unseen,covered,
      "all source regimes observed in train" if accepted else "unseen categorical mechanism regime")
