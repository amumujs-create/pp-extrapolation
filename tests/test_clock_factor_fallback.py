import numpy as np
import pytest
from pp_extrapolation.clock_factor_fallback import audit_clock,fit_clock,predict_clock


def data():
    t=np.tile(np.arange(1.,16.),3)
    origin=np.repeat([20.,25.,30.],15)
    return dict(x=np.column_stack((t,origin/30)),y=origin-t,groups=np.repeat(['a','b','c'],15))


def test_clock_audit_and_reject_nonaffine():
    r=data();a=audit_clock(r)
    assert a['slope']==-1 and a['max_residual']<1e-10
    with pytest.raises(ValueError):
        audit_clock({**r,'y':r['y']+.1*r['x'][:,0]**2})


def test_query_clock_changes_only_countdown_not_endpoint():
    tr=data();va={**tr,'groups':np.repeat('val',len(tr['y']))}
    f=fit_clock(tr,va,max_epochs=5)
    # Fix an interior endpoint so neither prediction is affected by clipping.
    import torch
    with torch.no_grad():
        f['model'][-1].weight.zero_()
        f['model'][-1].bias.fill_(.5)
    q=np.array([[1.,.8],[2.,.8]])
    p=predict_clock(f,q)
    assert abs((p[1]-p[0])+1.)<1e-5


def test_units_must_not_overlap():
    tr=data()
    with pytest.raises(ValueError):
        fit_clock(tr,tr,max_epochs=5)
