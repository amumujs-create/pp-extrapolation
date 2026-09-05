import numpy as np

from pp_extrapolation.support_gate import combine_support_gated, support_distance


def test_zero_beta_recovers_ungated_pp_and_large_distance_recovers_affine():
    affine = np.array([10.0, 10.0])
    correction = np.array([4.0, -4.0])
    distance = np.array([0.0, 100.0])
    baseline = combine_support_gated(
        affine, correction, distance, beta=0.0, output_cap=20.0
    )
    gated = combine_support_gated(
        affine, correction, distance, beta=1.0, output_cap=20.0
    )
    np.testing.assert_allclose(baseline, [14.0, 6.0])
    np.testing.assert_allclose(gated, [14.0, 10.0], atol=1e-10)


def test_distance_is_zero_inside_and_positive_beyond_train_interval():
    distance, audit = support_distance(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([[1.0], [3.0]]),
    )
    assert distance[0] == 0.0
    assert distance[1] > 0.0
    assert audit.outside_mask.tolist() == [False, True]
