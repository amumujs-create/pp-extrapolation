import numpy as np
import torch
from pp_extrapolation.temporal import DegradationContract,causal_history,TemporalPriorNet


def test_history_is_prefix_causal_and_unit_transform_invariant():
    t=np.arange(40,dtype=float);v=2.-.01*t
    contract=DegradationContract(1.4,'decreasing')
    full=causal_history(t,v,contract)
    prefix=causal_history(t[:20],v[:20],contract)
    for key in full:np.testing.assert_array_equal(full[key][:20],prefix[key])
    milliamps=causal_history(t,v*1000,DegradationContract(1400,'decreasing'))
    np.testing.assert_allclose(full['q'],milliamps['q'],atol=1e-6)
    np.testing.assert_allclose(full['prior'][10:],milliamps['prior'][10:],atol=1e-4)


def test_linear_trend_predicts_boundary_remaining_time():
    t=np.arange(40,dtype=float);v=2.-.01*t
    h=causal_history(t,v,DegradationContract(1.4,'decreasing'))
    np.testing.assert_allclose(h['prior'][10:,0],60-t[10:],atol=1e-4)
    inc=causal_history(t,10+.1*t,DegradationContract(20,'increasing'))
    np.testing.assert_allclose(inc['prior'][10:,0],100-t[10:],atol=1e-4)


def test_corrected_network_starts_at_online_prior():
    m=TemporalPriorNet(9);x=torch.randn(5,16,9)
    p=torch.rand(5,2);r=torch.softmax(torch.randn(5,2),1)
    y,g=m(x,p,r)
    torch.testing.assert_close(g,r)
    torch.testing.assert_close(y,(p*r).sum(1))
