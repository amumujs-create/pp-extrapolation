import numpy as np
import pytest
import torch
from pp_extrapolation.function_prior_sharing import (
    basis,integral,fit_bank,condition_gaussian,SharingRule,SharingFit,predict_sharing,
)


def rows():
    h=np.tile(np.linspace(.3,1.,12),3);g=np.repeat(['a','b','c'],12)
    level=np.repeat([2.,3.,4.],12)
    x=np.zeros((len(h),11));x[:,0]=h;x[:,6]=level
    return dict(x=x,y=h*level,rate=1/level,groups=g)


def test_positive_boundary_and_path_consistency():
    coef=torch.tensor([[[.3,.2,-.1],[.1,-.2,.1]]],dtype=torch.float64)
    m=torch.tensor([1.],dtype=torch.float64);mid=torch.tensor([.4],dtype=torch.float64)
    whole=integral(coef,m);upper=integral(coef,m,mid);lower=integral(coef,mid)
    torch.testing.assert_close(whole,upper+lower,atol=1e-10,rtol=1e-10)
    assert torch.all(whole>upper)
    torch.testing.assert_close(integral(coef,torch.zeros_like(m)),torch.zeros((1,2),dtype=torch.float64))


def test_rank_one_update_preserves_unobserved_directions():
    mean=torch.zeros((1,3),dtype=torch.float64);cov=torch.eye(3,dtype=torch.float64)[None]
    h=torch.tensor([.5],dtype=torch.float64)
    mu,s=condition_gaussian(mean,cov,h,torch.ones(1,dtype=torch.float64),.2,1.)
    phi=basis(h)[0];v=torch.tensor([-.5,1.,0.],dtype=torch.float64)
    assert abs(phi@v)<1e-12
    torch.testing.assert_close(v@s[0]@v,v@cov[0]@v)
    assert torch.linalg.matrix_rank(cov-s,tol=1e-8).item()==1
    assert torch.linalg.eigvalsh(s).min()>0
    assert mu.norm()>0


def test_unreliable_observation_does_not_update():
    m=torch.zeros((2,3),dtype=torch.float64);s=torch.eye(3,dtype=torch.float64).expand(2,-1,-1)
    a,b=condition_gaussian(m,s,torch.ones(2,dtype=torch.float64),torch.ones(2,dtype=torch.float64),1.,0.)
    torch.testing.assert_close(a,m);torch.testing.assert_close(b,s)


def test_fitted_prior_and_component_weights_psd():
    r=rows();bank=fit_bank(r,steps=20)
    assert np.linalg.eigvalsh(bank.covariances).min()>0
    for mode in ('component','tied','fixed','none'):
        model=SharingRule(mode);m,c,w=model.coefficient_distribution(bank,r['x'][:3],r['rate'][:3])
        assert torch.linalg.eigvalsh(c).min()>0
        if mode!='none':torch.testing.assert_close(w.sum(-1),torch.ones((3,3),dtype=torch.float64))
        if mode in ('tied','fixed'):torch.testing.assert_close(w[:,0],w[:,1])


def test_distribution_determinism_and_nonzero_uncertainty():
    r=rows();bank=fit_bank(r,steps=20);fit=SharingFit(SharingRule(),bank,{})
    p=predict_sharing(fit,r,samples=32);q=predict_sharing(fit,r,samples=32)
    np.testing.assert_array_equal(p['samples'],q['samples'])
    assert np.isfinite(p['samples']).all() and np.all(p['q95']>p['q05'])


def test_gradients_for_component_sharing():
    r=rows();bank=fit_bank(r,steps=20);model=SharingRule()
    p=model.draws(bank,r['x'][:4],r['rate'][:4],torch.randn(8,3,dtype=torch.float64))
    p.mean().backward()
    assert model.log_metric.grad is not None and torch.isfinite(model.log_metric.grad).all()


def test_invalid_rate_and_margin_rejected():
    r=rows()
    with pytest.raises(ValueError):fit_bank({**r,'rate':-r['rate']})
    bank=fit_bank(r,steps=5)
    with pytest.raises(ValueError):SharingRule().coefficient_distribution(bank,r['x'][:1],np.array([0.]))
