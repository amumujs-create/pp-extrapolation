"""Final 9-setting structure cards and structure-only selection.

The cards describe train structure only. ``select_ppx_from_structure`` then
applies the frozen Final priority so the selected executor matches the
published 9-setting portfolio without using validation or test losses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .paper_ppx import AutoPaperPPXDecision, ExecutorName, select_ppx_from_structure
from .ppx_contract_inference import OptionalExecutorDetection, detect_optional_executors

FinalSetting = Literal[
    "HUST",
    "Sunwoda",
    "N-CMAPSS",
    "Virkler",
    "RWTH",
    "MATR-b2",
    "MICH",
    "NASA",
    "MATR19",
]


@dataclass(frozen=True)
class FinalNineCard:
    name: FinalSetting
    final_executor: ExecutorName
    known_boundary: bool
    boundary_value: float | None
    unit_stable_regime: bool
    shifted_support: bool
    time_varying_condition: bool
    variable_history_length: bool = False


FINAL_NINE_CARDS: tuple[FinalNineCard, ...] = (
    FinalNineCard("HUST", "regime_transport", False, None, True, False, False),
    FinalNineCard("Sunwoda", "bounded", True, 0.8, False, False, False),
    FinalNineCard("N-CMAPSS", "history", False, None, False, False, True),
    FinalNineCard("Virkler", "unbounded", False, None, False, False, False),
    FinalNineCard("RWTH", "bounded", True, 0.8, False, False, False),
    FinalNineCard("MATR-b2", "regime_transport", False, None, True, False, False),
    FinalNineCard("MICH", "dual_scale", True, 0.8, False, True, False),
    FinalNineCard("NASA", "history", False, None, False, False, False, True),
    FinalNineCard("MATR19", "unbounded", False, None, False, False, False),
)


def final_nine_card(name: str) -> FinalNineCard:
    for card in FINAL_NINE_CARDS:
        if card.name == name:
            return card
    raise KeyError(name)


def _points(start: float, step: float, count: int) -> np.ndarray:
    return start + step * np.arange(count, dtype=float)


def make_final_nine_train(card: FinalNineCard) -> dict[str, np.ndarray]:
    """Build a tiny train table that matches one Final setting's structure."""
    if card.unit_stable_regime:
        units = np.repeat([f"u{i}" for i in range(4)], 3)
        cycles = np.tile(_points(1, 1, 3), 4)
        regime = np.repeat(["P1", "P1", "P2", "P2"], 3)
        health = np.tile(_points(1.0, -0.02, 3), 4)
    elif card.time_varying_condition:
        units = np.repeat(["e1", "e2", "e3"], 6)
        cycles = np.tile(_points(1, 1, 6), 3)
        regime = np.tile(["low", "high", "low", "high", "low", "high"], 3)
        health = np.tile(_points(1.0, -0.03, 6), 3)
    elif card.shifted_support:
        units = np.repeat(["c1", "c2", "c3"], 4)
        cycles = np.tile(_points(1, 1, 4), 3)
        regime = np.repeat(["same"], 12)
        health = np.concatenate(
            [
                _points(1.00, -0.02, 4),
                _points(0.70, -0.02, 4),
                _points(0.40, -0.02, 4),
            ]
        )
    elif card.variable_history_length:
        lengths = (3, 6, 9)
        units = np.concatenate([np.repeat(f"c{i+1}", n) for i, n in enumerate(lengths)])
        cycles = np.concatenate([_points(1, 1, n) for n in lengths])
        health = np.concatenate([np.linspace(1.0, 0.92, n) for n in lengths])
        regime = np.repeat(["same"], int(sum(lengths)))
    elif card.name == "Virkler":
        units = np.repeat(["crack1", "crack2"], 4)
        cycles = np.tile(_points(1, 1, 4), 2)
        regime = np.repeat(["same"], 8)
        health = np.tile(_points(0.1, 0.05, 4), 2)
    elif card.name == "MATR19":
        units = np.repeat(["b1", "b2"], 2)
        cycles = np.array([1, 2, 1, 2], dtype=float)
        regime = np.repeat(["same"], 4)
        health = np.array([0.95, 0.90, 0.94, 0.89])
    else:
        units = np.repeat(["c1", "c2", "c3"], 4)
        cycles = np.tile(_points(1, 1, 4), 3)
        regime = np.repeat(["same"], 12)
        health = np.tile(_points(1.0, -0.03, 4), 3)
    train: dict[str, np.ndarray] = {
        "groups": units,
        "cycles": cycles,
        "x": np.column_stack((health, cycles)),
        "y": np.ones(len(units)),
    }
    if card.unit_stable_regime or card.time_varying_condition:
        train["regime"] = regime
    return train


def replay_final_nine_selection(name: str) -> AutoPaperPPXDecision:
    card = final_nine_card(name)
    return select_ppx_from_structure(
        make_final_nine_train(card),
        boundary_value=card.boundary_value,
    )


def detect_final_nine(name: str) -> OptionalExecutorDetection:
    return detect_optional_executors(make_final_nine_train(final_nine_card(name)))
