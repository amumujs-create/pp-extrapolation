"""Online correction gate using only already-realized shadow forecasts."""
from __future__ import annotations

import numpy as np


def apply_causal_backtest_gate(
    candidate,
    anchor,
    truth,
    groups,
    origin_index,
    target_index,
    evaluate,
    *,
    required_wins=5,
):
    """Activate corrections after consecutive causal shadow-forecast wins."""
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    groups = np.asarray(groups)
    origin_index = np.asarray(origin_index, dtype=np.int64)
    target_index = np.asarray(target_index, dtype=np.int64)
    evaluate = np.asarray(evaluate, dtype=bool)
    if (
        candidate.ndim != 1
        or anchor.shape != candidate.shape
        or truth.shape != candidate.shape
        or groups.shape != candidate.shape
        or origin_index.shape != candidate.shape
        or target_index.shape != candidate.shape
        or evaluate.shape != candidate.shape
        or required_wins < 1
    ):
        raise ValueError("causal backtest inputs must be aligned vectors")
    if np.any(target_index <= origin_index):
        raise ValueError("every shadow target must follow its origin")

    prediction = anchor.copy()
    active = np.zeros(len(anchor), dtype=bool)
    available_counts = np.zeros(len(anchor), dtype=np.int64)
    winning_counts = np.zeros(len(anchor), dtype=np.int64)
    for label in np.unique(groups):
        rows = np.flatnonzero(groups == label)
        rows = rows[np.argsort(origin_index[rows], kind="stable")]
        for current in rows[evaluate[rows]]:
            eligible = rows[target_index[rows] <= origin_index[current]]
            chosen = []
            boundary = int(origin_index[current])
            for index in eligible[::-1]:
                if target_index[index] <= boundary:
                    chosen.append(index)
                    boundary = int(origin_index[index])
                    if len(chosen) == required_wins:
                        break
            available_counts[current] = len(chosen)
            if len(chosen) < required_wins:
                continue
            chosen = np.asarray(chosen, dtype=int)
            gain = (
                (truth[chosen] - anchor[chosen]) ** 2
                - (truth[chosen] - candidate[chosen]) ** 2
            )
            wins = int(np.sum(gain > 0))
            winning_counts[current] = wins
            if wins == required_wins and candidate[current] != anchor[current]:
                prediction[current] = candidate[current]
                active[current] = True
    return prediction, {
        "active": active,
        "available_shadow_count": available_counts,
        "winning_shadow_count": winning_counts,
    }


def apply_adaptive_causal_backtest_gate(
    candidate,
    anchor,
    truth,
    groups,
    origin_index,
    target_index,
    evaluate,
    *,
    minimum_history=2,
    maximum_history=5,
):
    """Activate after every available recent causal shadow forecast wins."""
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    groups = np.asarray(groups)
    origin_index = np.asarray(origin_index, dtype=np.int64)
    target_index = np.asarray(target_index, dtype=np.int64)
    evaluate = np.asarray(evaluate, dtype=bool)
    if (
        candidate.ndim != 1
        or anchor.shape != candidate.shape
        or truth.shape != candidate.shape
        or groups.shape != candidate.shape
        or origin_index.shape != candidate.shape
        or target_index.shape != candidate.shape
        or evaluate.shape != candidate.shape
        or minimum_history < 1
        or maximum_history < minimum_history
    ):
        raise ValueError("adaptive causal inputs must be aligned vectors")
    if np.any(target_index <= origin_index):
        raise ValueError("every shadow target must follow its origin")
    prediction = anchor.copy()
    active = np.zeros(len(anchor), dtype=bool)
    available_counts = np.zeros(len(anchor), dtype=np.int64)
    winning_counts = np.zeros(len(anchor), dtype=np.int64)
    for label in np.unique(groups):
        rows = np.flatnonzero(groups == label)
        rows = rows[np.argsort(origin_index[rows], kind="stable")]
        for current in rows[evaluate[rows]]:
            eligible = rows[target_index[rows] <= origin_index[current]]
            chosen = []
            boundary = int(origin_index[current])
            for index in eligible[::-1]:
                if target_index[index] <= boundary:
                    chosen.append(index)
                    boundary = int(origin_index[index])
                    if len(chosen) == maximum_history:
                        break
            available_counts[current] = len(chosen)
            if len(chosen) < minimum_history:
                continue
            chosen = np.asarray(chosen, dtype=int)
            gain = (
                (truth[chosen] - anchor[chosen]) ** 2
                - (truth[chosen] - candidate[chosen]) ** 2
            )
            wins = int(np.sum(gain > 0))
            winning_counts[current] = wins
            if wins == len(chosen) and candidate[current] != anchor[current]:
                prediction[current] = candidate[current]
                active[current] = True
    return prediction, {
        "active": active,
        "available_shadow_count": available_counts,
        "winning_shadow_count": winning_counts,
    }
