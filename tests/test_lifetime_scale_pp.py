from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pp_extrapolation import fit_lifetime_scale_pp,predict_lifetime_scale

def test_lifetime_identity_and_nonnegative_prediction():
    x=np.arange(1,31,dtype=np.float32)[:,None]; elapsed=x[:,0]; y=40-elapsed
    tr={"x":x[:20],"y":y[:20],"groups":np.repeat([0,1],10)}
    va={"x":x[20:25],"y":y[20:25],"groups":np.zeros(5)}
    f=fit_lifetime_scale_pp(tr,va,elapsed[:20],elapsed[20:25],max_epochs=30,patience=5,width=8)
    p=predict_lifetime_scale(f,x[25:],elapsed[25:])
    assert np.all(np.isfinite(p)) and np.all(p>=0)
    assert np.mean(np.abs(p-y[25:]))<5
