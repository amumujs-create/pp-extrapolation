"""Leakage-safe support grades and nested pseudo-extrapolation episodes.

All quantities used to grade an application row are frozen on source data.
Nested episodes keep the outer, inner, and fitting physical units disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


TailRule = str | Callable[[np.ndarray, float], np.ndarray]


def _readonly(value: object, *, dtype: object | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite non-empty vector or matrix")
    return result


def _vector(
    value: object, name: str, length: int, *, numeric: bool
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64 if numeric else object)
    if result.shape != (length,):
        raise ValueError(f"{name} must be a vector of length {length}")
    if numeric and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _coordinate_values(
    x: object, coordinate: object, length: int
) -> np.ndarray:
    if isinstance(coordinate, str):
        try:
            values = x[coordinate]  # type: ignore[index]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"coordinate column {coordinate!r} is unavailable"
            ) from error
        return _vector(values, "coordinate", length, numeric=True)
    if isinstance(coordinate, (int, np.integer)):
        features = _matrix(x, "x")
        index = int(coordinate)
        if not -features.shape[1] <= index < features.shape[1]:
            raise ValueError("coordinate column index is out of range")
        return features[:, index].copy()
    return _vector(coordinate, "coordinate", length, numeric=True)


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, mad)
    scale = np.where(scale > 1e-12, scale, standard)
    return center, np.where(scale > 1e-12, scale, 1.0)


def _ordered_unique(groups: np.ndarray) -> tuple[object, ...]:
    return tuple(
        sorted(
            set(groups.tolist()),
            key=lambda value: (type(value).__name__, repr(value)),
        )
    )


@dataclass(frozen=True)
class SupportGradeFit:
    """Robust source-only transform and physical-unit support summaries."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    coordinate_center: float
    coordinate_scale: float
    source_groups: tuple[object, ...]
    unit_centroids: np.ndarray
    distance_scale: float
    source_grade_max: float

    def __post_init__(self) -> None:
        center = _readonly(self.feature_center, dtype=np.float64)
        scale = _readonly(self.feature_scale, dtype=np.float64)
        centroids = _readonly(self.unit_centroids, dtype=np.float64)
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("feature center and scale must align")
        if (
            not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(
                "feature transform must be finite with positive scale"
            )
        if centroids.shape != (len(self.source_groups), len(center) + 1):
            raise ValueError(
                "unit centroids do not align with source groups and features"
            )
        numeric = (
            self.coordinate_center,
            self.coordinate_scale,
            self.distance_scale,
            self.source_grade_max,
        )
        if (
            not np.isfinite(numeric).all()
            or self.coordinate_scale <= 0
            or self.distance_scale <= 0
        ):
            raise ValueError(
                "support-grade scales and limits must be finite and valid"
            )
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "unit_centroids", centroids)

    @property
    def centroid_groups(self) -> tuple[object, ...]:
        return self.source_groups

    @property
    def centroids(self) -> np.ndarray:
        return self.unit_centroids


def _standardized(
    fit: SupportGradeFit, x: object, coordinate: object
) -> np.ndarray:
    features = _matrix(x, "x")
    if features.shape[1] != len(fit.feature_center):
        raise ValueError("x has the wrong feature dimension")
    ordered = _coordinate_values(x, coordinate, len(features))
    return np.column_stack(
        (
            (features - fit.feature_center) / fit.feature_scale,
            (ordered - fit.coordinate_center) / fit.coordinate_scale,
        )
    )


def _nearest_centroid_grade(
    fit: SupportGradeFit,
    standardized: np.ndarray,
    groups: np.ndarray | None,
) -> np.ndarray:
    distances = np.linalg.norm(
        standardized[:, None, :] - fit.unit_centroids[None, :, :], axis=2
    ) / np.sqrt(standardized.shape[1])
    if groups is not None:
        own = (
            groups[:, None]
            == np.asarray(fit.source_groups, dtype=object)[None, :]
        )
        if np.any(~np.any(~own, axis=1)):
            raise ValueError(
                "each training row requires a non-self source centroid"
            )
        distances = np.where(own, np.inf, distances)
    return np.min(distances, axis=1) / fit.distance_scale


def fit_support_grade(
    x: object,
    groups: Sequence[object],
    coordinate: object,
) -> SupportGradeFit:
    """Fit a robust source transform without retaining mutable source rows."""

    features = _matrix(x, "x")
    labels = _vector(groups, "groups", len(features), numeric=False)
    ordered = _coordinate_values(x, coordinate, len(features))
    unique = _ordered_unique(labels)
    if len(unique) < 2:
        raise ValueError("at least two physical source units are required")

    feature_center, feature_scale = _robust_scale(features)
    coordinate_center_array, coordinate_scale_array = _robust_scale(
        ordered[:, None]
    )
    coordinate_center = float(coordinate_center_array[0])
    coordinate_scale = float(coordinate_scale_array[0])
    standardized = np.column_stack(
        (
            (features - feature_center) / feature_scale,
            (ordered - coordinate_center) / coordinate_scale,
        )
    )
    centroids = np.stack(
        [np.median(standardized[labels == group], axis=0) for group in unique]
    )
    provisional = SupportGradeFit(
        feature_center,
        feature_scale,
        coordinate_center,
        coordinate_scale,
        unique,
        centroids,
        1.0,
        0.0,
    )
    raw_training = _nearest_centroid_grade(provisional, standardized, labels)
    distance_scale = max(float(np.quantile(raw_training, 0.90)), 1e-12)
    training_grades = raw_training / distance_scale
    return SupportGradeFit(
        feature_center,
        feature_scale,
        coordinate_center,
        coordinate_scale,
        unique,
        centroids,
        distance_scale,
        float(np.max(training_grades)),
    )


def transform_training_support_grades(
    fit: SupportGradeFit,
    x: object,
    groups: Sequence[object],
    coordinate: object,
) -> np.ndarray:
    """Grade source rows against frozen centroids excluding their own unit."""

    standardized = _standardized(fit, x, coordinate)
    labels = _vector(groups, "groups", len(standardized), numeric=False)
    unknown = set(labels.tolist()) - set(fit.source_groups)
    if unknown:
        raise ValueError(
            f"training groups were not present during fitting: {unknown}"
        )
    return _nearest_centroid_grade(fit, standardized, labels)


def transform_support_grade(
    fit: SupportGradeFit,
    x: object,
    coordinate: object,
) -> np.ndarray:
    """Pointwise application transform using only frozen source support."""

    standardized = _standardized(fit, x, coordinate)
    return _nearest_centroid_grade(fit, standardized, None)


@dataclass(frozen=True)
class Episode:
    """One nested, unit-disjoint pseudo-extrapolation episode."""

    outer_unit: object
    inner_unit: object
    cutoff: float
    fit_indices: np.ndarray
    tune_indices: np.ndarray
    query_indices: np.ndarray
    fit_units: tuple[object, ...]
    threshold: float

    def __post_init__(self) -> None:
        for name in ("fit_indices", "tune_indices", "query_indices"):
            values = _readonly(getattr(self, name), dtype=np.int64)
            if values.ndim != 1 or np.any(values < 0):
                raise ValueError(
                    f"{name} must contain nonnegative row indices"
                )
            object.__setattr__(self, name, values)
        if (
            self.outer_unit == self.inner_unit
            or self.outer_unit in self.fit_units
        ):
            raise ValueError("outer held unit leaked into nested support")
        if self.inner_unit in self.fit_units:
            raise ValueError("inner tune unit leaked into fit support")
        if not 0 < self.cutoff < 1 or not np.isfinite(self.threshold):
            raise ValueError("episode cutoff and threshold are invalid")

    @property
    def support_indices(self) -> np.ndarray:
        return self.fit_indices

    @property
    def outer_query_indices(self) -> np.ndarray:
        return self.query_indices

    @property
    def outer_held_unit(self) -> object:
        return self.outer_unit

    @property
    def inner_held_unit(self) -> object:
        return self.inner_unit

    @property
    def tuning_indices(self) -> np.ndarray:
        return self.tune_indices


@dataclass(frozen=True)
class NestedEpisodePlan:
    """Deterministic collection plus an explicit index-level leakage audit."""

    episodes: tuple[Episode, ...]
    groups: tuple[object, ...]
    cutoffs: tuple[float, ...]
    outer_rule: str
    inner_rule: str
    outer_exclusion_audit: tuple[bool, ...]
    same_unit_exclusion_audit: tuple[bool, ...]

    @property
    def audit_passed(self) -> bool:
        return all(self.outer_exclusion_audit) and all(
            self.same_unit_exclusion_audit
        )


def _tail_mask(
    rule: TailRule, values: np.ndarray, threshold: float
) -> np.ndarray:
    if callable(rule):
        mask = np.asarray(rule(values.copy(), threshold), dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("tail rule must return one boolean per row")
        return mask
    normalized = str(rule).lower().replace("-", "_")
    if normalized in {"high", "upper", "greater", "forward"}:
        return values > threshold
    if normalized in {"low", "lower", "less", "backward"}:
        return values < threshold
    raise ValueError("tail rules must be 'high'/'low' aliases or callables")


def _rule_name(rule: TailRule) -> str:
    return getattr(rule, "__name__", str(rule))


def build_nested_pseudo_extrapolation_episodes(
    groups: Sequence[object],
    coordinate: Sequence[float],
    outer_rule: TailRule = "high",
    inner_rule: TailRule = "high",
    *,
    cutoffs: Sequence[float] = (0.5, 0.65, 0.8),
) -> NestedEpisodePlan:
    """Build nested leave-two-units-out episodes from source-only thresholds.

    For every outer/inner unit pair, thresholds are fitted solely on all other
    units. Those units provide fit support, the inner unit provides tuning
    pseudo-tail rows, and the outer unit provides query rows.
    """

    labels = np.asarray(groups, dtype=object)
    ordered = _vector(coordinate, "coordinate", len(labels), numeric=True)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("groups must be a non-empty vector")
    unique = _ordered_unique(labels)
    if len(unique) < 3:
        raise ValueError(
            "nested episodes require at least three physical units"
        )
    depths = tuple(float(value) for value in cutoffs)
    if (
        not depths
        or not np.isfinite(depths).all()
        or any(not 0 < value < 1 for value in depths)
        or any(right <= left for left, right in zip(depths, depths[1:]))
    ):
        raise ValueError(
            "cutoffs must be strictly increasing finite values in (0, 1)"
        )

    episodes: list[Episode] = []
    outer_audit: list[bool] = []
    same_unit_audit: list[bool] = []
    for outer_unit in unique:
        for inner_unit in unique:
            if inner_unit == outer_unit:
                continue
            fit_units = tuple(
                unit
                for unit in unique
                if unit != outer_unit and unit != inner_unit
            )
            fit_pool = np.flatnonzero(np.isin(labels, fit_units))
            for cutoff in depths:
                threshold = float(np.quantile(ordered[fit_pool], cutoff))
                fit_tail = _tail_mask(inner_rule, ordered[fit_pool], threshold)
                fit_indices = fit_pool[~fit_tail]
                inner_rows = np.flatnonzero(labels == inner_unit)
                outer_rows = np.flatnonzero(labels == outer_unit)
                tune_indices = inner_rows[
                    _tail_mask(inner_rule, ordered[inner_rows], threshold)
                ]
                query_indices = outer_rows[
                    _tail_mask(outer_rule, ordered[outer_rows], threshold)
                ]
                if (
                    not len(fit_indices)
                    or not len(tune_indices)
                    or not len(query_indices)
                ):
                    continue
                episode = Episode(
                    outer_unit,
                    inner_unit,
                    cutoff,
                    fit_indices,
                    tune_indices,
                    query_indices,
                    fit_units,
                    threshold,
                )
                outer_ok = not np.any(
                    labels[np.concatenate((fit_indices, tune_indices))]
                    == outer_unit
                )
                same_unit_ok = (
                    set(labels[fit_indices].tolist()).isdisjoint(
                        set(labels[tune_indices].tolist())
                    )
                    and set(labels[fit_indices].tolist()).isdisjoint(
                        set(labels[query_indices].tolist())
                    )
                    and set(labels[tune_indices].tolist()).isdisjoint(
                        set(labels[query_indices].tolist())
                    )
                )
                if not outer_ok or not same_unit_ok:
                    raise RuntimeError(
                        "nested episode index leakage audit failed"
                    )
                episodes.append(episode)
                outer_audit.append(outer_ok)
                same_unit_audit.append(same_unit_ok)
    if not episodes:
        raise ValueError(
            "rules and cutoffs produced no non-empty nested episodes"
        )
    return NestedEpisodePlan(
        tuple(episodes),
        unique,
        depths,
        _rule_name(outer_rule),
        _rule_name(inner_rule),
        tuple(outer_audit),
        tuple(same_unit_audit),
    )


NestedEpisode = Episode
