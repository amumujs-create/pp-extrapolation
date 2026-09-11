"""Contract-certified, residual-only stochastic monotone transport.

The certificate implemented here is empirical and group-OOF: the structural
contract (the envelope) is exact, while the selected transport mass is only
certified on the supplied validation units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .risk_budgeted_prior import (
    apply_prior_residual,
    fit_support_scale,
    group_mse,
    local_budget_modulation,
    support_distance,
)


@dataclass(frozen=True)
class ContractEnvelope:
    """Frozen prior locations and positive outcome-space envelope radii."""

    location: np.ndarray
    radius: np.ndarray

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=np.float64).copy()
        radius = np.asarray(self.radius, dtype=np.float64).copy()
        if location.ndim != 1 or radius.shape != location.shape:
            raise ValueError("location and radius must be aligned vectors")
        if not np.all(np.isfinite(location)) or np.any(~np.isfinite(radius)):
            raise ValueError("contract envelope must be finite")
        if np.any(radius <= 0):
            raise ValueError("envelope radius must be positive")
        location.setflags(write=False)
        radius.setflags(write=False)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "radius", radius)

    def project_residual(self, residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=np.float64)
        return self.location[..., None] + self.radius[..., None] * np.tanh(
            residual / self.radius[..., None]
        )


@dataclass(frozen=True)
class TransportRiskDecision:
    rho: float
    mean_regret: float
    cvar_regret: float
    max_regret: float
    feasible_count: int


@dataclass(frozen=True)
class ResidualTransportFit:
    """Immutable fitted residual quantile map and train-only support state."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    residual_quantiles: np.ndarray
    probabilities: np.ndarray
    support: tuple[np.ndarray, np.ndarray, np.ndarray]
    distance_scale: float
    disagreement_scale: float
    rho: float
    risk_decision: TransportRiskDecision


def group_balanced_energy_score(
    y: np.ndarray, samples: np.ndarray, groups: np.ndarray
) -> float:
    """Equally weight physical units in the empirical CRPS/energy score."""
    y = np.asarray(y, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    groups = np.asarray(groups)
    if samples.ndim != 2 or y.shape != (len(samples),) or groups.shape != y.shape:
        raise ValueError("y, samples, and groups must align")
    first = np.mean(np.abs(samples - y[:, None]), axis=1)
    pairwise = np.mean(
        np.abs(samples[:, :, None] - samples[:, None, :]), axis=(1, 2)
    )
    row_score = first - 0.5 * pairwise
    return float(np.mean([
        np.mean(row_score[groups == label]) for label in np.unique(groups)
    ]))


def _group_energy(
    y: np.ndarray, samples: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    return np.asarray([
        group_balanced_energy_score(
            y[groups == label],
            samples[groups == label],
            groups[groups == label],
        )
        for label in np.unique(groups)
    ])


def _validate_baseline_envelope(
    samples: np.ndarray, envelope: ContractEnvelope
) -> None:
    lower = envelope.location[:, None] - envelope.radius[:, None]
    upper = envelope.location[:, None] + envelope.radius[:, None]
    if np.any(~np.isfinite(samples)) or np.any(samples < lower) or np.any(samples > upper):
        raise ValueError("baseline samples must be finite and inside the contract envelope")


def support_modulation(
    x: np.ndarray,
    disagreement: np.ndarray,
    model: ResidualTransportFit,
) -> np.ndarray:
    """Return train-support and disagreement modulation ``a(x)``."""
    distance = support_distance(np.asarray(x, dtype=np.float64), model.support)
    return local_budget_modulation(
        distance,
        np.asarray(disagreement, dtype=np.float64),
        model.distance_scale,
        model.disagreement_scale,
    )


def select_transport_rho(
    y: np.ndarray,
    groups: np.ndarray,
    baseline_samples: np.ndarray,
    transport_samples: np.ndarray,
    *,
    grid_size: int = 101,
    cvar_fraction: float = 0.20,
    mean_budget: float = 0.0,
    cvar_budget: float = 0.01,
    max_budget: float = 0.02,
) -> TransportRiskDecision:
    """Select the largest group-unit-safe mixture mass on OOF samples."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    baseline_samples = np.asarray(baseline_samples, dtype=np.float64)
    transport_samples = np.asarray(transport_samples, dtype=np.float64)
    if baseline_samples.shape != transport_samples.shape:
        raise ValueError("baseline and transport samples must have equal shape")
    if (
        baseline_samples.ndim != 2
        or len(baseline_samples) != len(y)
        or groups.shape != y.shape
    ):
        raise ValueError("outcomes, groups, and sample matrices must align")
    if grid_size < 2 or not 0 < cvar_fraction <= 1:
        raise ValueError("invalid risk selection settings")
    base = _group_energy(y, baseline_samples, groups)
    scale = max(float(np.mean(np.abs(base))), 1e-12)
    feasible: list[tuple[float, float, float, float]] = []
    for rho in np.linspace(0.0, 1.0, grid_size):
        # Deterministic quantile coupling makes rho=0 bitwise exact.
        mixed = (1.0 - rho) * baseline_samples + rho * transport_samples
        regret = (_group_energy(y, mixed, groups) - base) / scale
        count = max(1, int(np.ceil(cvar_fraction * len(regret))))
        summary = (
            float(np.mean(regret)),
            float(np.mean(np.sort(regret)[-count:])),
            float(np.max(regret)),
        )
        if (
            summary[0] <= mean_budget + 1e-12
            and summary[1] <= cvar_budget + 1e-12
            and summary[2] <= max_budget + 1e-12
        ):
            feasible.append((float(rho), *summary))
    rho, mean, cvar, maximum = feasible[-1]
    return TransportRiskDecision(rho, mean, cvar, maximum, len(feasible))


def _raw_transport_samples(
    model: ResidualTransportFit,
    x: np.ndarray,
    envelope: ContractEnvelope,
    disagreement: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(envelope.location):
        raise ValueError("x and envelope must align")
    z = (x - model.feature_center) / model.feature_scale
    conditional_mean = np.c_[np.ones(len(z)), z] @ model.coefficients
    quantile_residual = np.interp(
        uniforms.ravel(), model.probabilities, model.residual_quantiles
    ).reshape(uniforms.shape)
    modulation = support_modulation(x, disagreement, model)
    raw = modulation[:, None] * (
        conditional_mean[:, None] + quantile_residual
    )
    return envelope.project_residual(raw)


def fit_residual_transport(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    train_envelope: ContractEnvelope,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_groups: np.ndarray,
    validation_envelope: ContractEnvelope,
    validation_baseline_samples: np.ndarray,
    *,
    train_disagreement: np.ndarray | None = None,
    validation_disagreement: np.ndarray | None = None,
    quantile_count: int = 101,
    ridge: float = 1e-6,
) -> ResidualTransportFit:
    """Fit a monotone empirical residual transport and OOF risk certificate."""
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    train_groups = np.asarray(train_groups)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    validation_groups = np.asarray(validation_groups)
    if train_x.ndim != 2 or validation_x.ndim != 2:
        raise ValueError("features must be matrices")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if train_y.shape != train_envelope.location.shape or train_groups.shape != train_y.shape:
        raise ValueError("training arrays must align")
    if validation_y.shape != validation_envelope.location.shape:
        raise ValueError("validation arrays must align")
    if validation_groups.shape != validation_y.shape:
        raise ValueError("validation groups must align")
    validation_baseline_samples = np.asarray(
        validation_baseline_samples, dtype=np.float64
    )
    if (
        validation_baseline_samples.ndim != 2
        or len(validation_baseline_samples) != len(validation_y)
    ):
        raise ValueError("validation baseline samples must align")
    _validate_baseline_envelope(validation_baseline_samples, validation_envelope)
    center = np.median(train_x, axis=0)
    scale = np.quantile(train_x, 0.75, axis=0) - np.quantile(
        train_x, 0.25, axis=0
    )
    scale = np.where(scale > 1e-12, scale, np.std(train_x, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.c_[np.ones(len(train_x)), (train_x - center) / scale]
    target = train_y - train_envelope.location
    # Equal-unit weighted residual mean component.
    counts = {label: np.sum(train_groups == label) for label in np.unique(train_groups)}
    weights = np.asarray([1.0 / counts[label] for label in train_groups])
    gram = design.T @ (weights[:, None] * design) + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ (weights * target))
    innovation = target - design @ coefficients
    probabilities = np.linspace(0.0, 1.0, quantile_count)
    residual_quantiles = np.quantile(innovation, probabilities)
    support = fit_support_scale(train_x)
    train_disagreement = (
        np.zeros(len(train_x)) if train_disagreement is None
        else np.asarray(train_disagreement, dtype=np.float64)
    )
    validation_disagreement = (
        np.zeros(len(validation_x)) if validation_disagreement is None
        else np.asarray(validation_disagreement, dtype=np.float64)
    )
    if train_disagreement.shape != train_y.shape:
        raise ValueError("training disagreement must align")
    if validation_disagreement.shape != validation_y.shape:
        raise ValueError("validation disagreement must align")
    distance_scale = max(
        float(np.quantile(support_distance(train_x, support), 0.90)), 1.0
    )
    disagreement_scale = max(
        float(np.quantile(np.abs(train_disagreement), 0.90)), 1e-8
    )
    provisional = ResidualTransportFit(
        center.copy(), scale.copy(), coefficients, residual_quantiles,
        probabilities, support, distance_scale, disagreement_scale, 1.0,
        TransportRiskDecision(1.0, np.nan, np.nan, np.nan, 0),
    )
    sample_count = np.asarray(validation_baseline_samples).shape[1]
    uniforms = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    uniforms = np.broadcast_to(uniforms, (len(validation_x), sample_count))
    transported = _raw_transport_samples(
        provisional, validation_x, validation_envelope,
        validation_disagreement, uniforms,
    )
    decision = select_transport_rho(
        validation_y, validation_groups, validation_baseline_samples, transported
    )
    return ResidualTransportFit(
        center.copy(), scale.copy(), coefficients.copy(),
        residual_quantiles.copy(), probabilities.copy(), support,
        distance_scale, disagreement_scale, decision.rho, decision,
    )


def sample_residual_transport(
    model: ResidualTransportFit,
    x: np.ndarray,
    envelope: ContractEnvelope,
    baseline_samples: np.ndarray,
    *,
    disagreement: np.ndarray | None = None,
    sample_count: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample the certified distribution; rho zero exactly returns baseline."""
    baseline = np.asarray(baseline_samples, dtype=np.float64)
    if baseline.ndim != 2 or len(baseline) != len(envelope.location):
        raise ValueError("baseline samples and envelope must align")
    _validate_baseline_envelope(baseline, envelope)
    if model.rho == 0.0:
        return baseline.copy()
    count = baseline.shape[1] if sample_count is None else int(sample_count)
    if count != baseline.shape[1]:
        raise ValueError("sample_count must match baseline sample count")
    disagreement = (
        np.zeros(len(x)) if disagreement is None
        else np.asarray(disagreement, dtype=np.float64)
    )
    uniforms = ndtr(np.random.default_rng(seed).standard_normal((len(x), count)))
    transported = _raw_transport_samples(
        model, x, envelope, disagreement, uniforms
    )
    # Reuse the established bounded interpolation primitive row-by-row.
    return np.column_stack([
        apply_prior_residual(baseline[:, j], transported[:, j], model.rho)
        for j in range(count)
    ])


def predict_residual_mean(*args, **kwargs) -> np.ndarray:
    """Return the Monte-Carlo predictive mean."""
    return sample_residual_transport(*args, **kwargs).mean(axis=1)


def predict_residual_quantile(
    model: ResidualTransportFit,
    x: np.ndarray,
    envelope: ContractEnvelope,
    baseline_samples: np.ndarray,
    quantiles: float | np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Return predictive quantiles (first axis indexes requested quantiles)."""
    samples = sample_residual_transport(
        model, x, envelope, baseline_samples, **kwargs
    )
    return np.quantile(samples, quantiles, axis=1)
