"""Coverage-preserving credal conformal layer for PP-X.

The layer turns validation evidence about a PP correction into both a point
route and a set-valued authority.  A falsified correction is removed from the
point estimate, but retained as an epistemic endpoint of the prediction set.
Consequently the resulting interval contains the legacy PP conformal interval
by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .prior_falsification import (
    PriorFalsificationCertificate,
    certify_prior_falsification,
)


@dataclass(frozen=True)
class CredalConformalPPXFit:
    certificate: PriorFalsificationCertificate
    alpha: float
    residual_scale: float
    block_quantile: float
    n_calibration_units: int
    authority_lower: float
    authority_upper: float
    minimum_target: Optional[float]

    @property
    def approved(self) -> bool:
        return not self.certificate.falsified


@dataclass(frozen=True)
class CredalConformalPrediction:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    radius: np.ndarray


def _aligned_vectors(*values: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value) for value in values)
    if not arrays or arrays[0].ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays):
        raise ValueError("inputs must align")
    return arrays


def fit_credal_conformal_ppx(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    candidate: np.ndarray,
    support_distance: np.ndarray,
    *,
    alpha: float = 0.10,
    confidence: float = 0.95,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260913,
    minimum_target: Optional[float] = 0.0,
) -> CredalConformalPPXFit:
    """Fit a falsification decision and legacy-compatible block conformal radius.

    The authority set is ``{1}`` when PP is approved and ``[0, 1]`` when PP is
    falsified.  The latter keeps both fallback and PP as epistemic endpoints
    while deploying the fallback as the point prediction.
    """
    y, groups, fallback, candidate, support_distance = _aligned_vectors(
        y, groups, fallback, candidate, support_distance
    )
    y = y.astype(np.float64)
    fallback = fallback.astype(np.float64)
    candidate = candidate.astype(np.float64)
    support_distance = support_distance.astype(np.float64)
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    numeric = np.concatenate((y, fallback, candidate, support_distance))
    if not np.isfinite(numeric).all() or np.any(support_distance < 0):
        raise ValueError("numeric inputs must be finite and distance nonnegative")

    certificate = certify_prior_falsification(
        y,
        groups,
        fallback,
        candidate,
        confidence=confidence,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    residual_scale = max(float(np.median(np.abs(y - candidate))), 1e-8)
    local_scale = residual_scale * (1.0 + support_distance)
    score = np.abs(y - candidate) / local_scale
    labels = np.unique(groups)
    block_score = np.asarray([
        np.quantile(score[groups == label], 1.0 - alpha)
        for label in labels
    ])
    level = min(
        1.0,
        float(np.ceil((len(block_score) + 1) * (1.0 - alpha)))
        / max(len(block_score), 1),
    )
    quantile = (
        float(np.quantile(block_score, level, method="higher"))
        if len(block_score)
        else float("nan")
    )
    if certificate.falsified:
        authority_lower, authority_upper = 0.0, 1.0
    else:
        authority_lower = authority_upper = 1.0
    return CredalConformalPPXFit(
        certificate=certificate,
        alpha=float(alpha),
        residual_scale=residual_scale,
        block_quantile=quantile,
        n_calibration_units=len(labels),
        authority_lower=authority_lower,
        authority_upper=authority_upper,
        minimum_target=minimum_target,
    )


def legacy_candidate_interval(
    model: CredalConformalPPXFit,
    candidate: np.ndarray,
    support_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the legacy support-scaled PP interval used for comparison."""
    candidate, support_distance = _aligned_vectors(candidate, support_distance)
    candidate = candidate.astype(np.float64)
    support_distance = support_distance.astype(np.float64)
    if not np.isfinite(np.concatenate((candidate, support_distance))).all():
        raise ValueError("prediction inputs must be finite")
    if np.any(support_distance < 0):
        raise ValueError("support distance must be nonnegative")
    radius = model.block_quantile * model.residual_scale * (
        1.0 + support_distance
    )
    lower = candidate - radius
    if model.minimum_target is not None:
        lower = np.maximum(float(model.minimum_target), lower)
    return lower, candidate + radius


def predict_credal_conformal_ppx(
    model: CredalConformalPPXFit,
    fallback: np.ndarray,
    candidate: np.ndarray,
    support_distance: np.ndarray,
) -> CredalConformalPrediction:
    """Predict a safe point and a coverage-preserving credal interval."""
    fallback, candidate, support_distance = _aligned_vectors(
        fallback, candidate, support_distance
    )
    fallback = fallback.astype(np.float64)
    candidate = candidate.astype(np.float64)
    support_distance = support_distance.astype(np.float64)
    numeric = np.concatenate((fallback, candidate, support_distance))
    if not np.isfinite(numeric).all() or np.any(support_distance < 0):
        raise ValueError("prediction inputs must be finite and distance nonnegative")

    correction = candidate - fallback
    endpoint_a = fallback + model.authority_lower * correction
    endpoint_b = fallback + model.authority_upper * correction
    segment_lower = np.minimum(endpoint_a, endpoint_b)
    segment_upper = np.maximum(endpoint_a, endpoint_b)
    radius = model.block_quantile * model.residual_scale * (
        1.0 + support_distance
    )
    lower = segment_lower - radius
    if model.minimum_target is not None:
        lower = np.maximum(float(model.minimum_target), lower)
    upper = segment_upper + radius
    legacy_lower = candidate - radius
    if model.minimum_target is not None:
        legacy_lower = np.maximum(float(model.minimum_target), legacy_lower)
    lower = np.minimum(lower, legacy_lower)
    upper = np.maximum(upper, candidate + radius)
    point = candidate if model.approved else fallback
    return CredalConformalPrediction(point, lower, upper, radius)
