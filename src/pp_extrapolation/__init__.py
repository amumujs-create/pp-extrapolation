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
from .oof_uncertainty import (UncertaintyHeadFit, fit_uncertainty_head,
                              nested_oof_disagreement, predict_uncertainty)
from .regime_spline import (RegimeSplineFit, fit_regime_spline_pp, jacobian_diagnostics,
                            predict_regime_spline)
from .regime_mixture import (LatentRegimeFit,LatentRegimePPNet,
                             fit_latent_regime_pp,latent_regime_components,
                             predict_latent_regime)
from .residual_calibration import (ResidualGainSelection, combine_residual_gain,
                                   select_group_robust_residual_gain,
                                   select_residual_gain)
from .output_calibration import (OutputCalibrator, approve_dual_evidence, approve_transport,
                                 approve_seed_consensus, consensus_tail_probability,
                                 fit_output_calibrator, group_loo_affine_evidence,
                                 transport_direction_cosine,
                                 select_group_loo_calibrator)
from .presets import battery_dual_scale_pp_config

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
    "UncertaintyHeadFit",
    "nested_oof_disagreement",
    "fit_uncertainty_head",
    "predict_uncertainty",
    "RegimeSplineFit",
    "fit_regime_spline_pp",
    "predict_regime_spline",
    "jacobian_diagnostics",
    "LatentRegimeFit",
    "LatentRegimePPNet",
    "fit_latent_regime_pp",
    "predict_latent_regime",
    "latent_regime_components",
    "ResidualGainSelection",
    "combine_residual_gain",
    "select_residual_gain",
    "select_group_robust_residual_gain",
    "OutputCalibrator",
    "fit_output_calibrator",
    "select_group_loo_calibrator",
    "approve_seed_consensus",
    "consensus_tail_probability",
    "group_loo_affine_evidence",
    "approve_dual_evidence",
    "transport_direction_cosine",
    "approve_transport",
    "battery_dual_scale_pp_config",
]


from .priors import CounterfactualRays, PriorPairs, TransportTriples
from .temporal import DegradationContract, causal_history, fit_temporal, predict_temporal
from .boundary_quotient import (
    BoundaryQuotientFit,
    BoundaryQuotientPPNet,
    fit_boundary_quotient_pp,
    predict_boundary_affine,
    predict_boundary_quotient,
)

__all__ += [
    "BoundaryQuotientFit", "BoundaryQuotientPPNet",
    "fit_boundary_quotient_pp", "predict_boundary_affine",
    "predict_boundary_quotient",
]
