"""Cross-domain consensus residual head for stable PP-X prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConsensusRepresentation:
    ensemble: np.ndarray
    features: np.ndarray
    target_scale: float
    q90_seed_disagreement: float


@dataclass(frozen=True)
class CrossDomainConsensusHead:
    coefficient: np.ndarray
    intercept: float
    ridge_alpha: float
    training_domains: tuple[str, ...]


@dataclass(frozen=True)
class ConsensusPrediction:
    seeds: np.ndarray
    ensemble: np.ndarray
    correction: np.ndarray
    active_residual: bool
    q90_seed_disagreement: float


def consensus_representation(predictions: np.ndarray) -> ConsensusRepresentation:
    """Build affine-scale-free features from a PP-X seed ensemble."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[0] < 3:
        raise ValueError("predictions must have shape (at least 3 seeds, rows)")
    if predictions.shape[1] == 0 or not np.isfinite(predictions).all():
        raise ValueError("predictions must be finite and nonempty")
    ensemble = np.mean(predictions, axis=0)
    median = np.median(predictions, axis=0)
    trimmed = np.mean(np.sort(predictions, axis=0)[1:-1], axis=0)
    center = float(np.median(ensemble))
    target_scale = max(float(np.std(ensemble)), 1e-8)
    deviation = (predictions - ensemble) / target_scale
    row_disagreement = np.std(predictions, axis=0)
    coordinate = (ensemble - center) / target_scale
    features = np.column_stack((
        coordinate,
        coordinate**2,
        (median - ensemble) / target_scale,
        (trimmed - ensemble) / target_scale,
        row_disagreement / target_scale,
        np.ptp(predictions, axis=0) / target_scale,
        deviation.T,
    ))
    return ConsensusRepresentation(
        ensemble=ensemble,
        features=features,
        target_scale=target_scale,
        q90_seed_disagreement=float(
            np.quantile(row_disagreement, 0.90) / target_scale
        ),
    )


def normalized_residual_target(
    y: np.ndarray, representation: ConsensusRepresentation
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.shape != representation.ensemble.shape or not np.isfinite(y).all():
        raise ValueError("target must align with the ensemble")
    return (y - representation.ensemble) / representation.target_scale


def _domain_weights(domains: np.ndarray) -> np.ndarray:
    labels = np.unique(domains)
    weights = np.zeros(len(domains), dtype=np.float64)
    for label in labels:
        mask = domains == label
        weights[mask] = 1.0 / (len(labels) * np.sum(mask))
    return weights


def fit_cross_domain_consensus_head(
    features: np.ndarray,
    residual_target: np.ndarray,
    domains: np.ndarray,
    *,
    ridge_alpha: float = 100.0,
) -> CrossDomainConsensusHead:
    """Fit a domain-balanced normalized residual head."""
    features = np.asarray(features, dtype=np.float64)
    residual_target = np.asarray(residual_target, dtype=np.float64)
    domains = np.asarray(domains).astype(str)
    if (
        features.ndim != 2
        or residual_target.shape != (len(features),)
        or domains.shape != residual_target.shape
    ):
        raise ValueError("features, target, and domains must align")
    labels = np.unique(domains)
    if len(labels) < 3:
        raise ValueError("at least three training domains are required")
    if ridge_alpha <= 0 or not np.isfinite(features).all() or not np.isfinite(
        residual_target
    ).all():
        raise ValueError("training inputs and ridge alpha must be finite")
    weights = _domain_weights(domains)
    feature_mean = np.sum(weights[:, None] * features, axis=0)
    target_mean = float(np.sum(weights * residual_target))
    centered = features - feature_mean
    coefficient = np.linalg.solve(
        centered.T @ (weights[:, None] * centered)
        + float(ridge_alpha) * np.eye(features.shape[1]),
        centered.T @ (weights * (residual_target - target_mean)),
    )
    intercept = target_mean - float(feature_mean @ coefficient)
    return CrossDomainConsensusHead(
        coefficient=coefficient,
        intercept=intercept,
        ridge_alpha=float(ridge_alpha),
        training_domains=tuple(str(value) for value in labels),
    )


def predict_cross_domain_consensus(
    model: CrossDomainConsensusHead,
    predictions: np.ndarray,
    *,
    disagreement_cutoff: float = 0.12,
    residual_authority: float = 0.50,
    seed_shrinkage: float = 0.50,
) -> ConsensusPrediction:
    """Apply a gated meta-residual and contract the seed deviations."""
    if (
        disagreement_cutoff < 0
        or not 0 <= residual_authority <= 1
        or not 0 <= seed_shrinkage <= 1
    ):
        raise ValueError("invalid consensus policy")
    predictions = np.asarray(predictions, dtype=np.float64)
    representation = consensus_representation(predictions)
    if representation.features.shape[1] != len(model.coefficient):
        raise ValueError("seed count does not match the fitted head")
    active = representation.q90_seed_disagreement >= disagreement_cutoff
    if active:
        normalized = (
            representation.features @ model.coefficient + model.intercept
        )
        correction = (
            float(residual_authority)
            * representation.target_scale
            * normalized
        )
    else:
        correction = np.zeros_like(representation.ensemble)
    ensemble = representation.ensemble + correction
    seeds = ensemble + float(seed_shrinkage) * (
        predictions - representation.ensemble
    )
    return ConsensusPrediction(
        seeds=seeds,
        ensemble=ensemble,
        correction=correction,
        active_residual=bool(active),
        q90_seed_disagreement=representation.q90_seed_disagreement,
    )
