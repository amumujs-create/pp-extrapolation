import numpy as np
import pytest
from pp_extrapolation.output_calibration import OutputCalibrator,approve_dual_evidence,approve_seed_consensus,approve_transport,consensus_tail_probability,fit_output_calibrator,group_loo_affine_evidence,select_group_loo_calibrator,transport_direction_cosine

def test_affine_calibrator_recovers_scale_and_offset():
    prediction=np.arange(8,dtype=float);target=.7*prediction+4
    calibrator=fit_output_calibrator('affine',target,prediction)
    np.testing.assert_allclose(calibrator.predict(prediction),target,atol=1e-10)

def test_group_loo_selects_affine_for_consistent_miscalibration():
    prediction=np.arange(12,dtype=float);target=.6*prediction+5;groups=np.repeat(np.arange(4),3)
    calibrator,scores=select_group_loo_calibrator(target,prediction,groups,upper=20)
    assert calibrator.kind=='affine';assert scores['affine']<scores['identity']

def test_identity_remains_available():
    values=np.arange(12,dtype=float);groups=np.repeat(np.arange(4),3)
    calibrator,_=select_group_loo_calibrator(values,values,groups,upper=20)
    assert calibrator.kind=='identity'

def test_seed_consensus_requires_affine_from_every_seed():
    assert approve_seed_consensus([OutputCalibrator('affine') for _ in range(5)])
    assert not approve_seed_consensus([OutputCalibrator('affine'),OutputCalibrator('bias')])

def test_five_seed_consensus_is_exact_point_zero_five_sign_test():
    assert consensus_tail_probability(5,5)==pytest.approx(.03125)
    assert consensus_tail_probability(4,5)==pytest.approx(.1875)
    affine=[OutputCalibrator('affine') for _ in range(5)]
    assert approve_seed_consensus(affine,alpha=.05)
    assert not approve_seed_consensus(affine[:4]+[OutputCalibrator('identity')],alpha=.05)

def test_dual_evidence_requires_held_out_groups_and_seeds():
    prediction=np.arange(24,dtype=float);target=.6*prediction+5
    groups=np.repeat(np.arange(8),3)
    evidence=group_loo_affine_evidence(target,prediction,groups,upper=30)
    affine=[OutputCalibrator('affine') for _ in range(5)]
    assert evidence['wins']==8 and evidence['bootstrap_ci'][0]>0 and evidence['approved']
    assert approve_dual_evidence(affine,evidence)
    assert not approve_dual_evidence(affine,{**evidence,'approved':False})

def test_transport_rejects_opposite_extrapolation_ray():
    train=np.array([[0.,0.],[.1,-.1]])
    validation=np.array([[-1.,1.],[-1.,1.]])
    test=np.array([[1.,-1.],[1.,-1.]])
    cosine=transport_direction_cosine(train,validation,test)
    affine=[OutputCalibrator('affine') for _ in range(5)]
    assert cosine<-.99
    assert not approve_transport(affine,{'approved':True},direction_cosine=cosine)
