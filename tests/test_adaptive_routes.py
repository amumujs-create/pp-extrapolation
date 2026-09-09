import numpy as np
import pytest

from pp_extrapolation import (capacity_from_independent_groups,
                              inspection_boundary_quotient,
                              progress_quotient)


def test_progress_quotient_has_correct_odds_and_scale():
    elapsed=np.array([10.,20.]);progress=np.array([.5,.8])
    logit=np.log(progress/(1-progress))
    np.testing.assert_allclose(progress_quotient(elapsed,logit),[10.,5.])


def test_inspection_boundary_quotient_uses_offset_without_moving_observed_health():
    prediction=inspection_boundary_quotient(np.array([.4,.52]),np.array([.02,.01]),
        boundary=.5,inspection_offset=.02)
    np.testing.assert_allclose(prediction,[6.,0.])


def test_capacity_uses_independent_groups_and_is_capped():
    assert capacity_from_independent_groups(5)==8
    assert capacity_from_independent_groups(100,maximum_width=64)==64
    with pytest.raises(ValueError):capacity_from_independent_groups(0)
