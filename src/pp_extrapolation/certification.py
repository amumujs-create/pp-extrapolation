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
