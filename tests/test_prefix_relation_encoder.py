import numpy as np
from pp_extrapolation.prefix_relation_encoder import prefix_features,ridge_fit,fit_prefix,predict_prefix
from pp_extrapolation.function_prior_sharing import fit_bank


def rows():
    rng=np.random.default_rng(4);x=rng.normal(0,.1,(24,11));x[:,0]=np.tile(np.linspace(1,.3,8),3)
    return dict(x=x,cycles=np.tile(np.arange(8),3),groups=np.repeat(['a','b','c'],8),
                rate=np.ones(24)*.2,y=x[:,0]*2)


def test_features_are_causal_and_label_free():
    r=rows();z=prefix_features(r);changed={k:v.copy() for k,v in r.items()}
    changed['x'][7,2:]=100;changed['y'][:]=-999
    np.testing.assert_array_equal(z[:7],prefix_features(changed)[:7])
    changed['x']=r['x'].copy()
    np.testing.assert_array_equal(z,prefix_features(changed))


def test_predictions_do_not_read_labels():
    r=rows();fit=fit_prefix(r,fit_bank(r),steps=2)
    first=predict_prefix(fit,r,samples=16)
    changed={k:v.copy() for k,v in r.items()};changed['y'][:]=-1e8
    np.testing.assert_array_equal(first['mean'],predict_prefix(fit,changed,samples=16)['mean'])
    assert np.isfinite(first['mean']).all() and np.all(first['q95']>first['q05'])


def test_duplicate_cycle_rejected():
    r=rows();r['cycles'][1]=r['cycles'][0]
    import pytest
    with pytest.raises(ValueError):prefix_features(r)
