import pytest

from pp_extrapolation.contracts import (
    ContractSpec,
    TargetContract,
    admissible_experts,
    trajectory_contract,
)


def test_contract_experts_do_not_depend_on_dataset_identity():
    observed = trajectory_contract()
    shifted = trajectory_contract(condition_shift=True)
    assert "nonlinear_residual" in admissible_experts(observed)
    assert admissible_experts(shifted) == (
        "linear_rate",
        "damped_acceleration",
    )


def test_known_boundary_requires_finite_boundary():
    with pytest.raises(ValueError, match="boundary_value"):
        ContractSpec(
            TargetContract.KNOWN_BOUNDARY,
            ordered_coordinate=True,
            causal_history=False,
        )
    with pytest.raises(ValueError, match="finite"):
        ContractSpec(
            TargetContract.KNOWN_BOUNDARY,
            ordered_coordinate=True,
            causal_history=False,
            boundary_value=float("nan"),
        )


def test_condition_transfer_requires_regime_identifier():
    with pytest.raises(ValueError, match="regime identifier"):
        ContractSpec(
            TargetContract.CONDITION_TRANSFER,
            ordered_coordinate=True,
            causal_history=True,
        )
