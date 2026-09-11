"""Contract-conditioned strong-anchor prior residual for CCMR v2.3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_dynamics_bank import (
    CausalDynamicsBankFit,
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from .contracts import ContractSpec, admissible_experts
from .predictor_portfolio import (
    PredictorPortfolioFit,
    fit_predictor_portfolio,
    predict_predictor_portfolio,
)


@dataclass(frozen=True)
class ContractRiskExpertsFit:
    contract: ContractSpec
    portfolio: PredictorPortfolioFit
    prior_bank: CausalDynamicsBankFit
    feature_dimension: int


def _portfolio_features(correction, context):
    correction = np.asarray(correction, dtype=float)
    context = np.asarray(context, dtype=float)
    if correction.ndim != 2 or context.ndim != 2:
        raise ValueError("correction and context must be matrices")
    if len(correction) != len(context):
        raise ValueError("correction and context rows must align")
    return np.column_stack([context, correction])


def fit_contract_risk_experts(
    contract,
    train_correction,
    train_context,
    train_y,
    train_groups,
    validation_correction,
    validation_context,
    validation_y,
    validation_groups,
    *,
    include_engression=True,
    portfolio_seeds=(42, 43, 44),
    validation_mean_cap=0.0,
    validation_cvar_cap=0.01,
    validation_max_cap=0.02,
):
    """Fit the anchor first and a risk-bounded prior residual second."""
    if not isinstance(contract, ContractSpec):
        raise TypeError("contract must be a ContractSpec")
    train_features = _portfolio_features(train_correction, train_context)
    validation_features = _portfolio_features(
        validation_correction, validation_context
    )
    portfolio = fit_predictor_portfolio(
        train_features,
        train_y,
        train_groups,
        validation_features,
        validation_y,
        validation_groups,
        anchor_index=0,
        seeds=portfolio_seeds,
        include_engression=include_engression,
    )
    train_anchor = predict_predictor_portfolio(portfolio, train_features)
    validation_anchor = predict_predictor_portfolio(
        portfolio, validation_features
    )
    bank = fit_causal_dynamics_bank(
        train_correction,
        train_context,
        train_y,
        train_groups,
        train_anchor,
        validation_correction,
        validation_context,
        validation_y,
        validation_groups,
        validation_anchor,
        validation_mean_cap=validation_mean_cap,
        validation_cvar_cap=validation_cvar_cap,
        validation_max_cap=validation_max_cap,
        expert_names=admissible_experts(contract),
    )
    return ContractRiskExpertsFit(
        contract, portfolio, bank, train_features.shape[1]
    )


def predict_contract_risk_experts(model, correction, context):
    features = _portfolio_features(correction, context)
    if features.shape[1] != model.feature_dimension:
        raise ValueError("prediction features do not match fitted contract")
    anchor = predict_predictor_portfolio(model.portfolio, features)
    prediction, evidence = predict_causal_dynamics_bank(
        model.prior_bank, correction, context, anchor
    )
    evidence = {
        **evidence,
        "strong_anchor": anchor,
        "prior_delta": prediction - anchor,
    }
    return prediction, evidence
