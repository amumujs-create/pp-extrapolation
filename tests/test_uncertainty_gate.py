import numpy as np
import pytest
from pp_extrapolation import combine_uncertainty_gated

def test_uncertainty_and_distance_shrink_only_residual():
    a=np.array([[2.,2.],[2.,2.]]); r=np.array([[4.,4.],[4.,4.]])
    out=combine_uncertainty_gated(a,r,np.array([0.,1.]),np.array([0.,1.]),beta=1,gamma=1,output_cap=10)
    assert np.allclose(out[:,0],6)
    assert np.allclose(out[:,1],2+4*np.exp(-2))

def test_uncertainty_gate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        combine_uncertainty_gated(np.ones(2),np.ones(2),np.ones(2),np.ones(2),beta=1,gamma=1,output_cap=2)
