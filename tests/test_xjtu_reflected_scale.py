import numpy as np
from xjtu_reflected_scale_pp_v3 import reflected_scale
def test_opposite_ray_reflects_validation_scale_around_identity():
 truth=np.array([1.,2.,3.]);prediction=2*truth
 validation,test=reflected_scale(truth,prediction)
 assert np.isclose(validation,.5) and np.isclose(test,1.5)
