"""Reusable unit-disjoint strict-hull split for one ordered degradation coordinate."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _plain_list(values: np.ndarray) -> list:
    return [value.item() if isinstance(value, np.generic) else value for value in values]


def strict_hull_split_1d(
    frame: pd.DataFrame,
    *,
    unit_column: str,
    coordinate: str,
    train_cutoff: float,
    direction: str,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split units, then retain validation/test rows beyond train support.

    direction='low' means degradation moves below the training coordinate range;
    direction='high' means it moves above that range.
    """
    if direction not in {"low", "high"}:
        raise ValueError("direction must be 'low' or 'high'")
    units = np.asarray(sorted(frame[unit_column].dropna().unique(), key=str), dtype=object)
    if len(units) < 3:
        raise ValueError("at least three physical units are required")
    rng = np.random.default_rng(seed)
    units = units[rng.permutation(len(units))]
    n_train = max(1, int(np.floor(len(units) * train_fraction)))
    n_validation = max(1, int(np.floor(len(units) * validation_fraction)))
    if n_train + n_validation >= len(units):
        n_train = len(units) - 2
        n_validation = 1
    train_units = units[:n_train]
    validation_units = units[n_train : n_train + n_validation]
    test_units = units[n_train + n_validation :]

    train_pool = frame[frame[unit_column].isin(train_units)]
    if direction == "low":
        train = train_pool[train_pool[coordinate] >= train_cutoff].copy()
        boundary = float(train[coordinate].min())
        validation = frame[
            frame[unit_column].isin(validation_units) & (frame[coordinate] < boundary)
        ].copy()
        test = frame[
            frame[unit_column].isin(test_units) & (frame[coordinate] < boundary)
        ].copy()
    else:
        train = train_pool[train_pool[coordinate] <= train_cutoff].copy()
        boundary = float(train[coordinate].max())
        validation = frame[
            frame[unit_column].isin(validation_units) & (frame[coordinate] > boundary)
        ].copy()
        test = frame[
            frame[unit_column].isin(test_units) & (frame[coordinate] > boundary)
        ].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("the requested strict-hull split produced an empty partition")
    audit = {
        "seed": int(seed),
        "direction": direction,
        "train_support_boundary": boundary,
        "train_units": _plain_list(train_units),
        "validation_units": _plain_list(validation_units),
        "test_units": _plain_list(test_units),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    return train, validation, test, audit
