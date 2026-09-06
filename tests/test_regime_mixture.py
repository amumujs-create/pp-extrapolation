import numpy as np
import torch
from pp_extrapolation.regime_mixture import LatentRegimePPNet

def test_transition_probability_is_monotone_in_oriented_progress_at_fixed_context():
    model=LatentRegimePPNet(2,1.,0.)
    x=torch.tensor([[-1.,.2],[0.,.2],[1.,.2]])
    with torch.no_grad():g=model.components(x)[1].squeeze(1).numpy()
    assert np.all(np.diff(g)>=0)
