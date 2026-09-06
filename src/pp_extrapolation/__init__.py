"""PP strict-hull extrapolation package."""

from .hull import ConvexHullAudit, audit_convex_hull_support
from .gating import PriorGateDecision, select_prior_from_scores
from .certification import (ExtrapolationCertificate, RegimeCertificate,
                            certify_categorical_regime, certify_extrapolation)
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
from .support_gate import (
    SupportGateSelection,
    predict_support_gated,
    select_support_gate,
    support_distance,
)
from .shell_gate import (ShellCalibration, ShellEvidence, apply_shell_calibration,
                         calibrate_distance_shells)
from .uncertainty_gate import (UncertaintyGateSelection,
    combine_uncertainty_gated, component_ensemble, predict_uncertainty_gated,
    select_uncertainty_gate)

__all__ = [
    "ConvexHullAudit",
    "DegradationContract",
    "causal_history",
    "fit_temporal",
    "predict_temporal",
    "PriorPairs",
    "TransportTriples",
    "CounterfactualRays",
    "RegimeCertificate",
    "ExtrapolationCertificate",
    "certify_categorical_regime",
    "certify_extrapolation",
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
    "SupportGateSelection",
    "support_distance",
    "select_support_gate",
    "predict_support_gated",
    "ShellCalibration",
    "ShellEvidence",
    "calibrate_distance_shells",
    "apply_shell_calibration",
    "UncertaintyGateSelection",
    "combine_uncertainty_gated",
    "component_ensemble",
    "select_uncertainty_gate",
    "predict_uncertainty_gated",
]


from .priors import CounterfactualRays, PriorPairs, TransportTriples
from .temporal import DegradationContract, causal_history, fit_temporal, predict_temporal
