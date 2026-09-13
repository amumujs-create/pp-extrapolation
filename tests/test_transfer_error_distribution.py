import numpy as np
import pytest
from pp_extrapolation.transfer_error_distribution import distance_features,fit_error,factors


def fixture():
    x=np.column_stack((np.ones(12),np.linspace(0,1,12)))
    return fit_error(x,np.linspace(-.2,.2,12),np.repeat(['a','b','c'],4)),x


def test_spread_control_preserves_mean():
    model,x=fixture()
    np.testing.assert_allclose(factors(model,x,'spread_only').mean(1),1.)


def test_conditional_factors_positive_reproducible():
    model,x=fixture();a=factors(model,x)
    assert np.isfinite(a).all() and (a>0).all()
    np.testing.assert_array_equal(a,factors(model,x))


def test_invalid_errors_rejected():
    with pytest.raises(ValueError):fit_error(np.ones((3,2)),np.array([1.,np.nan,2.]),np.array(['a','b','c']))


def test_support_distance_direction():
    x=np.array([[.8],[.4]])
    np.testing.assert_allclose(distance_features(x,1.,.5)[:,1],[.4,1.2])
