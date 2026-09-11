"""Physical-unit certificate for falsifying harmful prior routes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriorFalsificationCertificate:
    mean_unit_log_regret: float
    upper_confidence_bound: float
    confidence: float
    n_units: int
    falsified: bool


def unit_log_regret(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Return log(RMSE_candidate / RMSE_fallback) per physical unit."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    fallback = np.asarray(fallback, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        y.ndim != 1
        or groups.shape != y.shape
        or fallback.shape != y.shape
        or candidate.shape != y.shape
    ):
        raise ValueError("y, groups, fallback, and candidate must align")
    values = []
    for label in np.unique(groups):
        mask = groups == label
        base = np.sqrt(np.mean((fallback[mask] - y[mask]) ** 2))
        trial = np.sqrt(np.mean((candidate[mask] - y[mask]) ** 2))
        values.append(np.log(max(float(trial), 1e-12) / max(float(base), 1e-12)))
    return np.asarray(values, dtype=np.float64)


def certify_prior_falsification(
    y: np.ndarray,
    groups: np.ndarray,
    fallback: np.ndarray,
    candidate: np.ndarray,
    *,
    confidence: float = 0.95,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260912,
) -> PriorFalsificationCertificate:
    """Falsify unless the one-sided mean-regret upper bound is negative."""
    if not 0.5 < confidence < 1 or bootstrap_replicates < 100:
        raise ValueError("invalid certificate settings")
    regret = unit_log_regret(y, groups, fallback, candidate)
    if len(regret) < 2:
        return PriorFalsificationCertificate(
            mean_unit_log_regret=float(np.mean(regret)),
            upper_confidence_bound=float("inf"),
            confidence=float(confidence),
            n_units=len(regret),
            falsified=True,
        )
    rng = np.random.default_rng(seed)
    index = rng.integers(
        0, len(regret), size=(int(bootstrap_replicates), len(regret))
    )
    means = np.mean(regret[index], axis=1)
    upper = float(np.quantile(means, confidence))
    return PriorFalsificationCertificate(
        mean_unit_log_regret=float(np.mean(regret)),
        upper_confidence_bound=upper,
        confidence=float(confidence),
        n_units=len(regret),
        falsified=upper >= 0.0,
    )
