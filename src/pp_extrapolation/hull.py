"""Train-only convex-hull membership audit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class ConvexHullAudit:
    outside_mask: np.ndarray
    distance: np.ndarray
    outside_fraction: float
    affine_rank: int

    def summary(self) -> dict:
        return {
            "criterion": "outside_convex_hull_of_train",
            "outside_count": int(self.outside_mask.sum()),
            "total_count": int(len(self.outside_mask)),
            "outside_fraction": float(self.outside_fraction),
            "affine_rank": int(self.affine_rank),
            "distance_median": float(np.median(self.distance)),
            "distance_max": float(np.max(self.distance)),
        }


def audit_convex_hull_support(
    x_train: np.ndarray,
    x_source: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    tolerance: float = 1e-8,
) -> ConvexHullAudit:
    train = np.asarray(x_train, dtype=np.float64)
    source = np.asarray(x_source, dtype=np.float64)
    if train.ndim != 2 or source.ndim != 2 or train.shape[1] != source.shape[1]:
        raise ValueError("train and source hull arrays must have shape (n, k)")
    if len(train) < 2 or len(source) < 1 or not np.isfinite(train).all() or not np.isfinite(source).all():
        raise ValueError("finite train/source rows are required")
    if feature_names is not None and len(feature_names) != train.shape[1]:
        raise ValueError("feature_names must match hull dimension")

    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_scaled = (train - mean) / scale
    source_scaled = (source - mean) / scale
    origin = train_scaled.mean(axis=0)
    centered = train_scaled - origin
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    reference = max(float(singular[0]) if len(singular) else 0.0, 1.0)
    rank = int(np.sum(singular > tolerance * reference))
    shifted = source_scaled - origin

    if rank == 0:
        affine_distance = np.linalg.norm(shifted, axis=1)
        facet_distance = np.zeros(len(source))
    else:
        basis = vt[:rank].T
        train_z = centered @ basis
        source_z = shifted @ basis
        affine_distance = np.linalg.norm(shifted - source_z @ basis.T, axis=1)
        if rank == 1:
            low, high = float(train_z[:, 0].min()), float(train_z[:, 0].max())
            facet_distance = np.maximum.reduce(
                [low - source_z[:, 0], source_z[:, 0] - high, np.zeros(len(source))]
            )
        else:
            try:
                hull = ConvexHull(np.unique(np.round(train_z, 12), axis=0))
            except QhullError as exc:
                raise ValueError(f"convex hull construction failed: {exc}") from exc
            normals, offsets = hull.equations[:, :-1], hull.equations[:, -1]
            signed = source_z @ normals.T + offsets
            facet_distance = np.maximum(
                signed / np.linalg.norm(normals, axis=1).clip(min=1e-12), 0.0
            ).max(axis=1)
    distance = np.hypot(affine_distance, facet_distance)
    outside = distance > tolerance
    return ConvexHullAudit(outside, distance, float(outside.mean()), rank)
