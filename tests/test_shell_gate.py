import numpy as np
import pytest
from pp_extrapolation import calibrate_distance_shells, apply_shell_calibration

def test_shells_accept_only_supported_better_stable_regions():
    d=np.r_[np.full(20,.25),np.full(20,.75),np.full(5,1.5)]
    y=np.linspace(0,1,len(d)); baseline=y+.2
    predictions=np.stack([y+.05,y-.05])
    predictions[:,20:40]=np.stack([y[20:40]+.3,y[20:40]+.3])
    calibration=calibrate_distance_shells(d,y,predictions,baseline,
        edges=(0,.5,1,2),minimum_count=20)
    assert [s.accepted for s in calibration.shells]==[True,False,False]
    assert apply_shell_calibration(calibration,np.array([.2,.7,1.2])).tolist()==[True,False,False]

def test_shell_gate_rejects_bad_shapes_and_unseen_distance():
    with pytest.raises(ValueError):
        calibrate_distance_shells(np.ones(3),np.ones(3),np.ones((3,)),np.ones(3))
    c=calibrate_distance_shells(np.full(20,.2),np.ones(20),np.ones((2,20)),np.ones(20)+1,
        edges=(0,.5,1),minimum_count=20)
    assert not apply_shell_calibration(c,np.array([.8]))[0]
