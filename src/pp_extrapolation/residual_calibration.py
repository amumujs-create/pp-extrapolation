"""Validation-only calibration of PP's neural residual confidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


@dataclass(frozen=True)
class ResidualGainSelection:
    """Chosen neural-residual gain and its validation audit."""

    gain: float
    candidates: tuple[dict, ...]


def select_group_robust_residual_gain(
    affine: np.ndarray,
    correction: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    *,
    output_cap: float,
    gains: Iterable[float] = DEFAULT_GAINS,
    minimum_relative_gain: float = 0.02,
    minimum_group_win_fraction: float = 0.8,
) -> ResidualGainSelection:
    """Approve a gain only when its benefit is consistent across validation units."""
    truth = np.asarray(truth, dtype=np.float64)
    groups = np.asarray(groups)
    gain_values = tuple(float(value) for value in gains)
    if 1.0 not in gain_values:
        raise ValueError("group-robust selection requires gain=1 baseline")
    predictions = {
        gain: combine_residual_gain(
            affine, correction, gain=gain, output_cap=output_cap
        )
        for gain in gain_values
    }
    baseline_error = (predictions[1.0] - truth) ** 2
    baseline_mse = float(np.mean(baseline_error))
    unique_groups = np.unique(groups)
    rows = []
    for gain in gain_values:
        error = (predictions[gain] - truth) ** 2
        mse = float(np.mean(error))
        relative_gain = (baseline_mse - mse) / max(baseline_mse, 1e-12)
        wins = 0
        for group in unique_groups:
            mask = groups == group
            if float(np.mean(error[mask])) < float(np.mean(baseline_error[mask])):
                wins += 1
        fraction = float(wins / max(len(unique_groups), 1))
        accepted = gain == 1.0 or (
            relative_gain >= minimum_relative_gain
            and fraction >= minimum_group_win_fraction
        )
        rows.append({"gain": gain, "validation_mse": mse,
                     "relative_gain_vs_one": relative_gain,
                     "group_win_fraction": fraction, "accepted": accepted})
    eligible = [row for row in rows if row["accepted"]]
    selected = min(eligible, key=lambda row: (row["validation_mse"], abs(row["gain"] - 1.0)))
    return ResidualGainSelection(float(selected["gain"]), tuple(rows))


def combine_residual_gain(
    affine: np.ndarray,
    correction: np.ndarray,
    *,
    gain: float,
    output_cap: float,
) -> np.ndarray:
    affine = np.asarray(affine, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if affine.shape != correction.shape:
        raise ValueError("affine and correction must have identical shapes")
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and nonnegative")
    return np.clip(affine + float(gain) * correction, 0.0, float(output_cap))


def select_residual_gain(
    affine: np.ndarray,
    correction: np.ndarray,
    truth: np.ndarray,
    *,
    output_cap: float,
    gains: Iterable[float] = DEFAULT_GAINS,
) -> ResidualGainSelection:
    """Select residual confidence from validation labels only."""
    truth = np.asarray(truth, dtype=np.float64)
    rows = []
    for gain in gains:
        prediction = combine_residual_gain(
            affine, correction, gain=float(gain), output_cap=output_cap
        )
        rows.append(
            {
                "gain": float(gain),
                "validation_mse": float(np.mean((prediction - truth) ** 2)),
            }
        )
    if not rows:
        raise ValueError("at least one gain is required")
    selected = min(rows, key=lambda row: (row["validation_mse"], abs(row["gain"] - 1.0)))
    return ResidualGainSelection(float(selected["gain"]), tuple(rows))
