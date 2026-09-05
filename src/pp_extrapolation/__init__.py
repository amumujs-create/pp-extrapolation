"""PP strict-hull extrapolation package."""

from .hull import ConvexHullAudit, audit_convex_hull_support
from .gating import PriorGateDecision, select_prior_from_scores
from .metrics import regression_metrics
from .model import (
    PPFit,
    PPNet,
    fit_feature_scale,
    fit_pp,
    predict,
    select_affine_initialization,
)
from .split import strict_hull_split_1d

__all__ = [
    "ConvexHullAudit",
    "PriorPairs",
    "TransportTriples",
    "PPFit",
    "PPNet",
    "PriorGateDecision",
    "audit_convex_hull_support",
    "fit_feature_scale",
    "fit_pp",
    "predict",
    "regression_metrics",
    "select_prior_from_scores",
    "select_affine_initialization",
    "strict_hull_split_1d",
]


from .priors import PriorPairs, TransportTriples
