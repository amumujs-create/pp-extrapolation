import numpy as np

from pp_extrapolation.residual_calibration import (
    combine_residual_gain,
    select_group_robust_residual_gain,
    select_residual_gain,
)


def test_gain_zero_recovers_affine_and_one_recovers_pp():
    affine = np.array([3.0, 5.0])
    correction = np.array([2.0, -1.0])
    np.testing.assert_allclose(
        combine_residual_gain(affine, correction, gain=0.0, output_cap=10.0), affine
    )
    np.testing.assert_allclose(
        combine_residual_gain(affine, correction, gain=1.0, output_cap=10.0),
        affine + correction,
    )


def test_validation_selection_chooses_known_residual_gain():
    affine = np.array([2.0, 4.0, 6.0])
    correction = np.array([4.0, -2.0, 2.0])
    truth = affine + 0.5 * correction
    chosen = select_residual_gain(
        affine, correction, truth, output_cap=20.0, gains=(0.0, 0.5, 1.0)
    )
    assert chosen.gain == 0.5
    assert chosen.candidates[1]["validation_mse"] == 0.0


def test_group_robust_selector_rejects_gain_that_hurts_one_of_two_groups():
    affine = np.array([0.0, 0.0, 0.0, 0.0])
    correction = np.ones(4)
    truth = np.array([1.5, 1.5, 0.5, 0.5])
    groups = np.array(["a", "a", "b", "b"])
    chosen = select_group_robust_residual_gain(
        affine, correction, truth, groups, output_cap=3.0,
        gains=(1.0, 1.5), minimum_relative_gain=0.02,
        minimum_group_win_fraction=0.8,
    )
    assert chosen.gain == 1.0
