"""PP strict-hull extrapolation package."""

from .hull import ConvexHullAudit, audit_convex_hull_support
from .gating import PriorGateDecision, select_prior_from_scores
from .certification import (ExtrapolationCertificate, RegimeCertificate,
                            certify_categorical_regime, certify_extrapolation)
from .metrics import regression_metrics
from .generalization_policy import PriorTrustDecision, select_prior_trust
from .adaptive_shrinkage import (
    AdaptiveShrinkageDecision,
    apply_adaptive_shrinkage,
    select_adaptive_prior_shrinkage,
)
from .residual_authority import (
    CrossfitAuthorityResult,
    ResidualAuthorityPolicy,
    apply_residual_authority,
    crossfit_residual_authority,
    distance_shell_edges,
    fit_residual_authority,
)
from .continuous_portfolio import (
    ContinuousPortfolioDecision,
    combine_portfolio,
    select_continuous_portfolio,
)
from .stability_first import (
    StabilityFirstDecision,
    raw_unit_regret,
    regret_summary,
    select_stability_first_prior,
    unit_regret,
)
from .regime_prior_bank import (
    AutoRegimePriorDecision,
    RegimeMap,
    fit_auto_regime_prior,
    fit_regime_map,
    predict_auto_regime_prior,
    regime_probabilities,
)
from .invariant_residual import (
    InvariantResidualFit,
    fit_invariant_residual,
    predict_invariant_residual,
)
from .consensus_residual import (
    ConsensusResidualFit,
    fit_consensus_residual,
    predict_consensus_residual,
)
from .causal_backtest import apply_causal_backtest_gate
from .causal_dynamics_bank import (
    CausalDynamicsBankFit,
    DynamicsExpert,
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from .small_cohort_route import (
    SmallCohortRouteDecision,
    select_ccmr_v22_route,
)
from .paper_ppx import (
    PPXCandidateEvidence,
    PPXContract,
    PaperPPXDecision,
    admissible_executors,
    select_paper_ppx,
)
from .regime_router import (
    RegimeRouteDecision,
    select_ccmr_regime_route,
    select_ccmr_v19_regime_route,
)
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
from .presets import (adaptive_shrinkage_policy_config,
                      battery_dual_scale_pp_config,
                      continuous_portfolio_policy_config,
                      robust_generalization_policy_config,
                      safety_continuation_pp_config,
                      safety_continuation_trust_grid)
from .adaptive_routes import (capacity_from_independent_groups,
                              inspection_boundary_quotient,
                              progress_quotient)

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
    "PriorTrustDecision",
    "select_prior_trust",
    "AdaptiveShrinkageDecision",
    "apply_adaptive_shrinkage",
    "select_adaptive_prior_shrinkage",
    "ResidualAuthorityPolicy",
    "CrossfitAuthorityResult",
    "distance_shell_edges",
    "fit_residual_authority",
    "crossfit_residual_authority",
    "apply_residual_authority",
    "ContinuousPortfolioDecision",
    "combine_portfolio",
    "select_continuous_portfolio",
    "StabilityFirstDecision",
    "unit_regret",
    "raw_unit_regret",
    "regret_summary",
    "select_stability_first_prior",
    "RegimeMap",
    "AutoRegimePriorDecision",
    "fit_regime_map",
    "regime_probabilities",
    "fit_auto_regime_prior",
    "predict_auto_regime_prior",
    "InvariantResidualFit",
    "fit_invariant_residual",
    "predict_invariant_residual",
    "ConsensusResidualFit",
    "fit_consensus_residual",
    "predict_consensus_residual",
    "apply_causal_backtest_gate",
    "DynamicsExpert",
    "CausalDynamicsBankFit",
    "fit_causal_dynamics_bank",
    "predict_causal_dynamics_bank",
    "SmallCohortRouteDecision",
    "select_ccmr_v22_route",
    "PPXContract",
    "PPXCandidateEvidence",
    "PaperPPXDecision",
    "admissible_executors",
    "select_paper_ppx",
    "RegimeRouteDecision",
    "select_ccmr_regime_route",
    "select_ccmr_v19_regime_route",
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
    "adaptive_shrinkage_policy_config",
    "continuous_portfolio_policy_config",
    "robust_generalization_policy_config",
    "safety_continuation_pp_config",
    "safety_continuation_trust_grid",
    "capacity_from_independent_groups",
    "inspection_boundary_quotient",
    "progress_quotient",
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
from .log_boundary_quotient import (
    LogBoundaryQuotientFit,
    fit_log_boundary_quotient_pp,
    predict_log_boundary_affine,
    predict_log_boundary_quotient,
)

__all__ += [
    "BoundaryQuotientFit", "BoundaryQuotientPPNet",
    "fit_boundary_quotient_pp", "predict_boundary_affine",
    "predict_boundary_quotient",
    "LogBoundaryQuotientFit", "fit_log_boundary_quotient_pp",
    "predict_log_boundary_affine", "predict_log_boundary_quotient",
]
from .lifetime_scale import fit_lifetime_scale_pp, predict_lifetime_scale, LifetimeScaleFit
from .latent_pp import fit_latent_pp, predict_latent, LatentFit
from .transferability_gate import PriorEvidence, GateDecision, select_ppx_route
from .executor_policy import ExecutorEvidence, ExecutorDecision, select_residual_executor

__all__ += ["PriorEvidence", "GateDecision", "select_ppx_route"]
