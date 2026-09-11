"""Grade-CVaR Implicit Expert (GCIE).

The support grade is a label-free, frozen distance to source-unit centroids.
Source rows use leave-one-unit-out (LOO) centroids; application rows use every
source centroid.  The implicit generator is trained with group-balanced energy
score, group-CVaR risk, and an ordered-coordinate monotonicity penalty.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _matrix(value: object, name: str, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    if dimension is not None and result.shape[1] != dimension:
        raise ValueError(f"{name} has the wrong feature dimension")
    return result


def _vector(value: object, name: str, length: int) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (length,):
        raise ValueError(f"{name} must be a vector of length {length}")
    return result


@dataclass(frozen=True)
class SupportGradeFit:
    """Frozen robust feature scale and source physical-unit centroids."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    centroid_groups: np.ndarray
    centroids: np.ndarray
    nearest_centroids: int
    distance_scale: float

    def __post_init__(self) -> None:
        center = _readonly(np.asarray(self.feature_center, dtype=np.float64))
        scale = _readonly(np.asarray(self.feature_scale, dtype=np.float64))
        groups = _readonly(np.asarray(self.centroid_groups))
        centroids = _readonly(np.asarray(self.centroids, dtype=np.float64))
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("support feature center and scale must align")
        if np.any(~np.isfinite(center)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("support feature scale must be finite and positive")
        if centroids.shape != (len(groups), len(center)) or len(groups) < 2:
            raise ValueError("at least two aligned source centroids are required")
        if not np.isfinite(centroids).all():
            raise ValueError("source centroids must be finite")
        if not 1 <= int(self.nearest_centroids) < len(groups):
            raise ValueError("nearest_centroids must allow LOO grading")
        if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive")
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "centroid_groups", groups)
        object.__setattr__(self, "centroids", centroids)


def _centroid_distance(
    fit: SupportGradeFit,
    x: np.ndarray,
    *,
    row_groups: np.ndarray | None,
) -> np.ndarray:
    standardized = (x - fit.feature_center) / fit.feature_scale
    distances = np.linalg.norm(
        standardized[:, None, :] - fit.centroids[None, :, :], axis=2
    ) / math.sqrt(x.shape[1])
    if row_groups is not None:
        same = row_groups[:, None] == fit.centroid_groups[None, :]
        if np.any(np.sum(~same, axis=1) < fit.nearest_centroids):
            raise ValueError("not enough non-self centroids for LOO support grade")
        distances = np.where(same, np.inf, distances)
    nearest = np.partition(
        distances, fit.nearest_centroids - 1, axis=1
    )[:, : fit.nearest_centroids]
    return nearest.mean(axis=1)


def fit_support_grade(
    x: np.ndarray,
    groups: Sequence[object],
    *,
    nearest_centroids: int = 1,
) -> tuple[SupportGradeFit, np.ndarray]:
    """Fit source centroids and return mandatory LOO source-row grades."""
    features = _matrix(x, "x")
    labels = _vector(groups, "groups", len(features))
    unique = np.asarray(list(dict.fromkeys(labels.tolist())))
    if len(unique) < 2:
        raise ValueError("support grading requires at least two physical units")
    if not 1 <= int(nearest_centroids) < len(unique):
        raise ValueError("nearest_centroids must be in [1, n_source_units)")
    center = np.median(features, axis=0)
    q25, q75 = np.quantile(features, (0.25, 0.75), axis=0)
    scale = q75 - q25
    fallback = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - center) / scale
    centroids = np.stack(
        [standardized[labels == group].mean(axis=0) for group in unique]
    )
    provisional = SupportGradeFit(
        center, scale, unique, centroids, int(nearest_centroids), 1.0
    )
    raw_loo = _centroid_distance(provisional, features, row_groups=labels)
    distance_scale = max(float(np.quantile(raw_loo, 0.90)), 1e-12)
    fitted = SupportGradeFit(
        center, scale, unique, centroids, int(nearest_centroids), distance_scale
    )
    return fitted, (raw_loo / distance_scale).astype(np.float32)


def support_grade(
    fit: SupportGradeFit,
    x: np.ndarray,
    *,
    row_groups: Sequence[object] | None = None,
) -> np.ndarray:
    """Compute frozen grades; ``row_groups`` requests source-row LOO grading."""
    features = _matrix(x, "x", len(fit.feature_center))
    labels = (
        None
        if row_groups is None
        else _vector(row_groups, "row_groups", len(features))
    )
    raw = _centroid_distance(fit, features, row_groups=labels)
    return (raw / fit.distance_scale).astype(np.float32)


@dataclass(frozen=True)
class GCIEConfig:
    hidden_dims: tuple[int, ...] = (64, 64)
    noise_dim: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 500
    patience: int = 70
    sample_count: int = 8
    cvar_fraction: float = 0.34
    cvar_weight: float = 0.5
    monotonic_weight: float = 1.0
    nearest_centroids: int = 1
    minimum_scale: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.noise_dim <= 0 or self.sample_count < 2 or self.sample_count % 2:
            raise ValueError("noise_dim must be positive and sample_count must be even")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.max_epochs < 1 or self.patience < 0:
            raise ValueError("invalid early-stopping settings")
        if not 0 < self.cvar_fraction <= 1:
            raise ValueError("cvar_fraction must lie in (0, 1]")
        if self.cvar_weight < 0 or self.monotonic_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.nearest_centroids < 1 or self.minimum_scale <= 0:
            raise ValueError("grade and scale settings must be positive")


def fixed_antithetic_noise(
    rows: int, sample_count: int, noise_dim: int, seed: int
) -> torch.Tensor:
    """Return deterministic paired ``z, -z`` standard-normal samples."""
    if rows < 1 or sample_count < 2 or sample_count % 2 or noise_dim < 1:
        raise ValueError("invalid antithetic noise shape")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    half = torch.randn(
        rows, sample_count // 2, noise_dim, generator=generator, dtype=torch.float32
    )
    return torch.cat((half, -half), dim=1)


class GCIEGenerator(nn.Module):
    """Bounded implicit generator with separate location and positive scale."""

    def __init__(self, input_dim: int, config: GCIEConfig):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim + 1
        for width in config.hidden_dims:
            layers.extend((nn.Linear(current, int(width)), nn.SiLU()))
            current = int(width)
        self.trunk = nn.Sequential(*layers)
        self.location_head = nn.Linear(current, 1)
        self.scale_head = nn.Linear(current, 1)
        noise_width = int(config.hidden_dims[-1])
        self.noise_net = nn.Sequential(
            nn.Linear(current + config.noise_dim, noise_width),
            nn.SiLU(),
            nn.Linear(noise_width, 1),
        )
        self.minimum_scale = float(config.minimum_scale)

    def components(
        self, x_grade: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x_grade)
        location = self.location_head(hidden).squeeze(1)
        scale = F.softplus(self.scale_head(hidden).squeeze(1)) + self.minimum_scale
        expanded = hidden[:, None, :].expand(-1, noise.shape[1], -1)
        residual = torch.tanh(self.noise_net(torch.cat((expanded, noise), dim=2)).squeeze(2))
        return location, scale, residual

    def forward(self, x_grade: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        location, scale, residual = self.components(x_grade, noise)
        return torch.sigmoid(location[:, None] + scale[:, None] * residual)

    def location(self, x_grade: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(x_grade)
        return torch.sigmoid(self.location_head(hidden).squeeze(1))


def group_energy_risks(
    target: torch.Tensor, samples: torch.Tensor, group_index: torch.Tensor
) -> torch.Tensor:
    """Differentiable energy score, reduced separately for each group."""
    if samples.ndim != 2 or target.shape != (len(samples),):
        raise ValueError("target and samples must align")
    if group_index.shape != target.shape:
        raise ValueError("group_index must align with target")
    first = torch.mean(torch.abs(samples - target[:, None]), dim=1)
    pairwise = torch.mean(
        torch.abs(samples[:, :, None] - samples[:, None, :]), dim=(1, 2)
    )
    rows = first - 0.5 * pairwise
    return torch.stack(
        [rows[group_index == group].mean() for group in torch.unique(group_index)]
    )


def grade_cvar_energy_loss(
    target: torch.Tensor,
    samples: torch.Tensor,
    group_index: torch.Tensor,
    *,
    cvar_fraction: float,
    cvar_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    risks = group_energy_risks(target, samples, group_index)
    count = max(1, int(math.ceil(float(cvar_fraction) * len(risks))))
    cvar = torch.topk(risks, count).values.mean()
    balanced = risks.mean()
    return balanced + float(cvar_weight) * cvar, balanced, cvar


def monotonic_violation_penalty(
    mean_prediction: torch.Tensor,
    groups: Sequence[object],
    ordered_coordinate: Sequence[float],
) -> torch.Tensor:
    """Penalize increases as the within-unit cycle coordinate increases."""
    labels = np.asarray(groups)
    coordinate = np.asarray(ordered_coordinate, dtype=np.float64)
    if labels.shape != (len(mean_prediction),) or coordinate.shape != labels.shape:
        raise ValueError("monotonic inputs must align")
    if not np.isfinite(coordinate).all():
        raise ValueError("ordered_coordinate must be finite")
    terms = []
    for group in dict.fromkeys(labels.tolist()):
        indices = np.flatnonzero(labels == group)
        order = indices[np.argsort(coordinate[indices], kind="stable")]
        if len(order) > 1:
            values = mean_prediction[torch.as_tensor(order, dtype=torch.long)]
            terms.append(torch.relu(values[1:] - values[:-1]).mean())
    return (
        torch.stack(terms).mean()
        if terms
        else mean_prediction.new_zeros(())
    )


@dataclass
class GCIEFit:
    model: GCIEGenerator
    support: SupportGradeFit
    target_cap: float
    config: GCIEConfig
    selection: dict


def _prepared_input(
    x: np.ndarray, grade: np.ndarray, support: SupportGradeFit
) -> torch.Tensor:
    standardized = (x - support.feature_center) / support.feature_scale
    value = np.column_stack((standardized, grade))
    return torch.as_tensor(value, dtype=torch.float32)


def fit_gcie(
    train_x: np.ndarray,
    train_y: Sequence[float],
    train_groups: Sequence[object],
    train_coordinate: Sequence[float],
    validation_x: np.ndarray,
    validation_y: Sequence[float],
    validation_groups: Sequence[object],
    validation_coordinate: Sequence[float],
    *,
    config: GCIEConfig = GCIEConfig(),
) -> GCIEFit:
    """Fit GCIE and restore the best validation checkpoint."""
    tx = _matrix(train_x, "train_x")
    vx = _matrix(validation_x, "validation_x", tx.shape[1])
    ty = np.asarray(_vector(train_y, "train_y", len(tx)), dtype=np.float64)
    vy = np.asarray(_vector(validation_y, "validation_y", len(vx)), dtype=np.float64)
    tg = _vector(train_groups, "train_groups", len(tx))
    vg = _vector(validation_groups, "validation_groups", len(vx))
    tc = np.asarray(_vector(train_coordinate, "train_coordinate", len(tx)), dtype=np.float64)
    vc = np.asarray(
        _vector(validation_coordinate, "validation_coordinate", len(vx)), dtype=np.float64
    )
    if not all(np.isfinite(value).all() for value in (ty, vy, tc, vc)):
        raise ValueError("targets and ordered coordinates must be finite")
    if np.any(ty < 0) or np.any(vy < 0):
        raise ValueError("GCIE requires nonnegative targets")
    target_cap = max(float(np.max(ty)), 1.0)
    if np.any(ty > target_cap):
        raise ValueError("training targets violate the target contract")
    support, train_grade = fit_support_grade(
        tx, tg, nearest_centroids=config.nearest_centroids
    )
    validation_grade = support_grade(support, vx)
    train_value = _prepared_input(tx, train_grade, support)
    validation_value = _prepared_input(vx, validation_grade, support)
    train_target = torch.as_tensor(ty / target_cap, dtype=torch.float32)
    validation_target = torch.as_tensor(vy / target_cap, dtype=torch.float32)
    _, train_inverse = np.unique(tg, return_inverse=True)
    _, validation_inverse = np.unique(vg, return_inverse=True)
    train_index = torch.as_tensor(train_inverse, dtype=torch.long)
    validation_index = torch.as_tensor(validation_inverse, dtype=torch.long)
    train_noise = fixed_antithetic_noise(
        len(tx), config.sample_count, config.noise_dim, config.seed + 1000
    )
    validation_noise = fixed_antithetic_noise(
        len(vx), config.sample_count, config.noise_dim, config.seed + 2000
    )

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    model = GCIEGenerator(tx.shape[1], config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    def objective(
        value: torch.Tensor,
        target: torch.Tensor,
        index: torch.Tensor,
        groups: np.ndarray,
        coordinate: np.ndarray,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        samples = model(value, noise)
        energy, balanced, cvar = grade_cvar_energy_loss(
            target,
            samples,
            index,
            cvar_fraction=config.cvar_fraction,
            cvar_weight=config.cvar_weight,
        )
        monotonic = monotonic_violation_penalty(samples.mean(1), groups, coordinate)
        loss = energy + float(config.monotonic_weight) * monotonic
        return loss, {
            "energy": float(balanced.detach()),
            "cvar_energy": float(cvar.detach()),
            "monotonic_penalty": float(monotonic.detach()),
        }

    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    last_epoch = 0
    for epoch in range(1, int(config.max_epochs) + 1):
        last_epoch = epoch
        model.train()
        loss, train_terms = objective(
            train_value, train_target, train_index, tg, tc, train_noise
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite GCIE training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0, error_if_nonfinite=True
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss, validation_terms = objective(
                validation_value,
                validation_target,
                validation_index,
                vg,
                vc,
                validation_noise,
            )
        current = float(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(loss.detach()),
                "validation_objective": current,
                **{f"train_{key}": value for key, value in train_terms.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_terms.items()
                },
            }
        )
        if current < best_loss - 1e-10:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch > int(config.patience):
            break
    model.load_state_dict(best_state)
    model.eval()
    return GCIEFit(
        model,
        support,
        target_cap,
        config,
        {
            "config": asdict(config),
            "selected_epoch": best_epoch,
            "epochs_executed": last_epoch,
            "best_validation_objective": best_loss,
            "loss_history": history,
            "train_grade_summary": {
                "minimum": float(np.min(train_grade)),
                "median": float(np.median(train_grade)),
                "maximum": float(np.max(train_grade)),
                "self_unit_excluded": True,
            },
            "validation_grade_summary": {
                "minimum": float(np.min(validation_grade)),
                "median": float(np.median(validation_grade)),
                "maximum": float(np.max(validation_grade)),
                "source_centroids_only": True,
            },
        },
    )


def _prediction_inputs(fit: GCIEFit, x: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    features = _matrix(x, "x", len(fit.support.feature_center))
    grade = support_grade(fit.support, features)
    return features, _prepared_input(features, grade, fit.support)


def predict_samples(
    fit: GCIEFit,
    x: np.ndarray,
    *,
    sample_count: int | None = None,
    seed: int = 0,
    rho: float = 1.0,
    fallback: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Draw bounded samples; ``rho=0`` returns the anchor bitwise exactly."""
    features, value = _prediction_inputs(fit, x)
    count = fit.config.sample_count if sample_count is None else int(sample_count)
    if count < 2 or count % 2:
        raise ValueError("sample_count must be an even integer >= 2")
    if not np.isfinite(rho) or not 0 <= float(rho) <= 1:
        raise ValueError("rho must lie in [0, 1]")
    fit.model.eval()
    with torch.no_grad():
        location = (
            fit.model.location(value).cpu().numpy().astype(np.float64)
            * fit.target_cap
        )
    if fallback is None:
        anchor = np.repeat(location[:, None], count, axis=1)
    else:
        fallback_array = np.asarray(fallback)
        if fallback_array.shape == (len(features),):
            anchor = np.repeat(fallback_array[:, None], count, axis=1)
        elif fallback_array.shape == (len(features), count):
            anchor = fallback_array
        else:
            raise ValueError("fallback must be row-aligned points or samples")
        if not np.isfinite(anchor).all():
            raise ValueError("fallback must be finite")
    if float(rho) == 0.0:
        return anchor.copy()
    noise = fixed_antithetic_noise(
        len(features), count, fit.config.noise_dim, int(seed)
    )
    with torch.no_grad():
        generated = (
            fit.model(value, noise).cpu().numpy().astype(np.float64)
            * fit.target_cap
        )
    mixed = anchor + float(rho) * (generated - anchor)
    return np.clip(mixed, 0.0, fit.target_cap)


def predict_mean(fit: GCIEFit, x: np.ndarray, **kwargs: object) -> np.ndarray:
    """Return the antithetic Monte-Carlo predictive mean."""
    return predict_samples(fit, x, **kwargs).mean(axis=1)
