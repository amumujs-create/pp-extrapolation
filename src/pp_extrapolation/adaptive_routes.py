"""Reusable structural routes for heterogeneous extrapolation contracts."""
from __future__ import annotations

import numpy as np


def progress_quotient(elapsed, logit_progress, *, logit_clip: float = 12.0):
    """Convert scale-free progress logits to nonnegative remaining life."""
    elapsed = np.asarray(elapsed, dtype=np.float64)
    logit = np.asarray(logit_progress, dtype=np.float64)
    if elapsed.shape != logit.shape or np.any(elapsed < 0) or not np.isfinite(elapsed).all():
        raise ValueError("elapsed and logit_progress must be finite aligned arrays")
    if not np.isfinite(logit).all() or not np.isfinite(logit_clip) or logit_clip <= 0:
        raise ValueError("logit values and positive clip must be finite")
    return elapsed * np.exp(-np.clip(logit, -float(logit_clip), float(logit_clip)))


def inspection_boundary_quotient(health, rate, *, boundary: float,
                                 inspection_offset: float = 0.0,
                                 minimum_rate: float = 1e-6):
    """RUL prior for a known boundary observed at discrete inspections."""
    health = np.asarray(health, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    values = np.asarray([boundary, inspection_offset, minimum_rate], dtype=float)
    if health.shape != rate.shape or not np.isfinite(health).all() or not np.isfinite(rate).all():
        raise ValueError("health and rate must be finite aligned arrays")
    if not np.isfinite(values).all() or minimum_rate <= 0:
        raise ValueError("boundary parameters must be finite and minimum_rate positive")
    return np.maximum(float(boundary)+float(inspection_offset)-health, 0.0) / np.maximum(rate, minimum_rate)


def capacity_from_independent_groups(group_count: int, *, maximum_width: int = 64) -> int:
    """Choose a conservative power-of-two residual width from group evidence."""
    if int(group_count) != group_count or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    if int(maximum_width) != maximum_width or maximum_width < 2:
        raise ValueError("maximum_width must be an integer >=2")
    width = int(2 ** np.ceil(np.log2(max(2, int(group_count)))))
    return min(width, int(maximum_width))
