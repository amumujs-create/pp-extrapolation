import numpy as np
from pp_extrapolation import fit_uncertainty_head,predict_uncertainty
import pp_extrapolation.oof_uncertainty as oof

def test_uncertainty_head_is_nonnegative_and_single_model():
    x=np.linspace(-1,1,24)[:,None]
    split={'x':x,'y':np.zeros(24),'groups':np.repeat(np.arange(6),4)}
    target=np.abs(x[:,0])*.1
    head=fit_uncertainty_head(split,target,max_epochs=10)
    prediction=predict_uncertainty(head,x)
    assert prediction.shape==(24,)
    assert np.isfinite(prediction).all()
    assert (prediction>=0).all()

def test_nested_targets_never_train_on_held_out_group(monkeypatch):
    groups=np.repeat(np.arange(6),2)
    split={'x':np.column_stack((np.arange(12),groups)),'y':np.arange(12.),'groups':groups}
    seen=[]
    class Fit: target_scale=1.
    monkeypatch.setattr(oof,'select_affine_initialization',lambda tr,va:{})
    def fake_fit(tr,va,**kwargs):
        train_groups=set(tr['groups']);val_groups=set(va['groups'])
        assert train_groups.isdisjoint(val_groups)
        seen.append(train_groups|val_groups);return Fit()
    monkeypatch.setattr(oof,'fit_pp',fake_fit)
    monkeypatch.setattr(oof,'predict_components',lambda fit,x:(np.zeros(len(x)),x[:,0].astype(float)))
    target=oof.nested_oof_disagreement(split,outer_folds=3,teacher_seeds=(1,2),max_epochs=1)
    assert np.isfinite(target).all()
    # Each group is absent from at least one teacher training/validation universe.
    assert all(any(group not in universe for universe in seen) for group in range(6))
