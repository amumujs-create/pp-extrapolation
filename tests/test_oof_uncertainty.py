import numpy as np
from pp_extrapolation import fit_uncertainty_head,predict_uncertainty

def test_uncertainty_head_is_nonnegative_and_single_model():
    x=np.linspace(-1,1,24)[:,None]
    split={'x':x,'y':np.zeros(24),'groups':np.repeat(np.arange(6),4)}
    target=np.abs(x[:,0])*.1
    head=fit_uncertainty_head(split,target,max_epochs=10)
    prediction=predict_uncertainty(head,x)
    assert prediction.shape==(24,)
    assert np.isfinite(prediction).all()
    assert (prediction>=0).all()
