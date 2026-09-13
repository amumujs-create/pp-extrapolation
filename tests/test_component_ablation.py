import numpy as np
import torch
from pp_extrapolation.component_ablation import TiedComponentGate,fit_ablation
from pp_extrapolation.component_transfer import ComponentGate,fit_components,predict_components


def data():
    rng=np.random.default_rng(9);n=24
    e=dict(c=rng.normal(size=(n,3)),z=rng.normal(size=(n,8)),base=np.ones(n)*10,
           y=np.ones(n)*11,groups=np.repeat(['a','b','c'],8))
    v={**e,'groups':np.repeat('val',n)}
    return e,v


def test_shared_gate_same_allocated_parameters():
    full=ComponentGate();scalar=TiedComponentGate()
    assert sum(p.numel() for p in full.parameters())==sum(p.numel() for p in scalar.parameters())==147
    a=scalar(torch.randn(7,8))
    torch.testing.assert_close(a[:,0],a[:,1]);torch.testing.assert_close(a[:,1],a[:,2])
    a.sum().backward()
    assert all(p.grad is not None for p in scalar.parameters())


def test_component_reimplementation_matches_frozen_algorithm():
    e,v=data()
    old=fit_components([e],v,seed=42,max_epochs=20)
    new=fit_ablation([e],v,seed=42,max_epochs=20)
    assert old.selection['selected_epoch']==new.selection['selected_epoch']
    np.testing.assert_array_equal(predict_components(old,v['c'],v['z'],v['base']),predict_components(new,v['c'],v['z'],v['base']))
    for name,p in old.model.state_dict().items():
        torch.testing.assert_close(p,new.model.state_dict()[name],atol=0,rtol=0)


def test_shared_gate_cannot_select_canceling_components():
    gate=TiedComponentGate()(torch.zeros(3,8))
    components=torch.tensor([[1.,-1.,0.]]).expand(3,-1)
    torch.testing.assert_close((gate*components).sum(1),torch.zeros(3))
    # Functional distinction only, not evidence of real-data superiority.


def test_seed_determinism():
    e,v=data();a=fit_ablation([e],v,seed=43,shared=True,max_epochs=10);b=fit_ablation([e],v,seed=43,shared=True,max_epochs=10)
    np.testing.assert_array_equal(predict_components(a,v['c'],v['z'],v['base']),predict_components(b,v['c'],v['z'],v['base']))
