"""Leave-one-domain-out regret-bound head for residual authority."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FalsificationAuthorityHead:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    ridge_alpha: float
    upper_residual_quantile: float
    authority_scale: float
    confidence: float
    training_domains: tuple[str, ...]
    inner_oof_rmse: float
    inner_oof_upper_coverage: float


def _domain_weights(domains: np.ndarray) -> np.ndarray:
    domains = np.asarray(domains)
    weights = np.zeros(len(domains), dtype=np.float64)
    labels = np.unique(domains)
    for label in labels:
        mask = domains == label
        weights[mask] = 1.0 / (len(labels) * max(int(np.sum(mask)), 1))
    return weights


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=np.float64)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cumulative, quantile, side="left")])


def _ridge(x, y, weights, alpha):
    total = float(np.sum(weights))
    x_mean = np.sum(weights[:, None] * x, axis=0) / total
    y_mean = float(np.sum(weights * y) / total)
    centered = x - x_mean
    coefficient = np.linalg.solve(
        centered.T @ (weights[:, None] * centered)
        + float(alpha) * np.eye(x.shape[1]),
        centered.T @ (weights * (y - y_mean)),
    )
    return coefficient, y_mean - float(x_mean @ coefficient)


def fit_falsification_authority_head(
    features: np.ndarray,
    regret: np.ndarray,
    domains: np.ndarray,
    *,
    ridge_alphas: tuple[float, ...] = (
        0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
    ),
    confidence: float = 0.95,
) -> FalsificationAuthorityHead:
    """Fit a mean-regret head and calibrate its one-sided OOF upper bound."""
    features = np.asarray(features, dtype=np.float64)
    regret = np.asarray(regret, dtype=np.float64)
    domains = np.asarray(domains).astype(str)
    if (
        features.ndim != 2
        or regret.shape != (len(features),)
        or domains.shape != regret.shape
    ):
        raise ValueError("features, regret, and domains do not align")
    if len(np.unique(domains)) < 3:
        raise ValueError("at least three domains are required")
    if not np.isfinite(np.concatenate([features.ravel(), regret])).all():
        raise ValueError("training values must be finite")
    if not 0.5 < confidence < 1 or not ridge_alphas:
        raise ValueError("invalid head settings")

    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (features - center) / scale
    weights = _domain_weights(domains)
    labels = np.unique(domains)
    candidates = []
    for alpha in ridge_alphas:
        oof = np.empty_like(regret)
        for held_out in labels:
            keep = domains != held_out
            coefficient, intercept = _ridge(
                z[keep], regret[keep], _domain_weights(domains[keep]), alpha
            )
            oof[~keep] = z[~keep] @ coefficient + intercept
        mse = float(np.sum(weights * (oof - regret) ** 2))
        candidates.append((mse, float(alpha), oof))
    mse, alpha, oof = min(candidates, key=lambda row: (row[0], row[1]))
    residual = regret - oof
    upper_quantile = _weighted_quantile(residual, weights, confidence)
    coverage = float(np.sum(weights * (regret <= oof + upper_quantile)))
    authority_scale = max(
        _weighted_quantile(np.abs(regret), weights, 0.75), 1e-8
    )
    coefficient, intercept = _ridge(z, regret, weights, alpha)
    return FalsificationAuthorityHead(
        center=center,
        scale=scale,
        coefficient=coefficient,
        intercept=float(intercept),
        ridge_alpha=alpha,
        upper_residual_quantile=upper_quantile,
        authority_scale=authority_scale,
        confidence=float(confidence),
        training_domains=tuple(str(value) for value in labels),
        inner_oof_rmse=float(np.sqrt(mse)),
        inner_oof_upper_coverage=coverage,
    )


def predict_regret_upper(
    model: FalsificationAuthorityHead, features: np.ndarray
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(model.center):
        raise ValueError("features do not match fitted head")
    z = (features - model.center) / model.scale
    return (
        z @ model.coefficient
        + model.intercept
        + model.upper_residual_quantile
    )


def predict_residual_authority(
    model: FalsificationAuthorityHead, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous authority and its predicted regret upper bound."""
    upper = predict_regret_upper(model, features)
    authority = np.clip(-upper / model.authority_scale, 0.0, 1.0)
    return authority, upper


def apply_unit_authority(
    fallback: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    unit_labels: np.ndarray,
    authority: np.ndarray,
) -> np.ndarray:
    """Apply one outcome-free authority value to each physical unit."""
    fallback = np.asarray(fallback, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    unit_labels = np.asarray(unit_labels).astype(str)
    authority = np.asarray(authority, dtype=np.float64)
    if (
        fallback.ndim != 1
        or candidate.shape != fallback.shape
        or groups.shape != fallback.shape
        or authority.shape != unit_labels.shape
        or np.any((authority < 0) | (authority > 1))
    ):
        raise ValueError("authority inputs do not align")
    mapping = dict(zip(unit_labels, authority))
    if set(np.unique(groups)) - set(mapping):
        raise ValueError("authority is missing a physical unit")
    row_authority = np.asarray([mapping[value] for value in groups])
    return fallback + row_authority * (candidate - fallback)
