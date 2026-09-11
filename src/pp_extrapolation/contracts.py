"""Data-contract definitions for CCMR v2.3 AC-CRPE.

Contracts are declared from the target and deployment specification.  Dataset
names and observed test performance are deliberately absent from this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TargetContract(str, Enum):
    OBSERVED_TRAJECTORY = "observed_trajectory"
    LATENT_RUL = "latent_rul"
    CONDITION_TRANSFER = "condition_transfer"
    KNOWN_BOUNDARY = "known_boundary"


@dataclass(frozen=True)
class ContractSpec:
    target: TargetContract
    ordered_coordinate: bool
    causal_history: bool
    regime_identifier: bool = False
    boundary_value: float | None = None

    def __post_init__(self):
        if not self.ordered_coordinate:
            raise ValueError("AC-CRPE requires an ordered extrapolation coordinate")
        if self.target in (
            TargetContract.OBSERVED_TRAJECTORY,
            TargetContract.LATENT_RUL,
        ) and not self.causal_history:
            raise ValueError(f"{self.target.value} requires causal history")
        if (
            self.target == TargetContract.CONDITION_TRANSFER
            and not self.regime_identifier
        ):
            raise ValueError("condition transfer requires a regime identifier")
        if self.target == TargetContract.KNOWN_BOUNDARY:
            if self.boundary_value is None:
                raise ValueError("known-boundary contract requires boundary_value")
            if not float("-inf") < float(self.boundary_value) < float("inf"):
                raise ValueError("boundary_value must be finite")
        elif self.boundary_value is not None:
            raise ValueError("boundary_value is only valid for known boundary")


_ADMISSIBLE = {
    TargetContract.OBSERVED_TRAJECTORY: (
        "linear_rate",
        "damped_acceleration",
        "monotone_hinge",
        "nonlinear_residual",
    ),
    TargetContract.LATENT_RUL: (
        "linear_rate",
        "damped_acceleration",
        "monotone_hinge",
    ),
    TargetContract.CONDITION_TRANSFER: (
        "linear_rate",
        "damped_acceleration",
    ),
    TargetContract.KNOWN_BOUNDARY: (
        "linear_rate",
        "monotone_hinge",
    ),
}


def admissible_experts(contract: ContractSpec) -> tuple[str, ...]:
    """Return the expert family fixed by the target contract."""
    return _ADMISSIBLE[contract.target]


def trajectory_contract(*, condition_shift: bool = False) -> ContractSpec:
    """Build the contract used by the v2.3 trajectory development track."""
    if condition_shift:
        return ContractSpec(
            TargetContract.CONDITION_TRANSFER,
            ordered_coordinate=True,
            causal_history=True,
            regime_identifier=True,
        )
    return ContractSpec(
        TargetContract.OBSERVED_TRAJECTORY,
        ordered_coordinate=True,
        causal_history=True,
    )
