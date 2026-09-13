"""RAVEN-X v2 remediation: explicit sticky no-transition prior."""
import numpy as np
import torch
from torch import nn
from .regime_validity_attention import (RegimeValidityAttention,subset_prepared,
    PreparedWindows,mixture_energy)


class StickyRegimeValidityAttention(RegimeValidityAttention):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            self.route.weight.zero_();self.route.bias.zero_();self.route.bias[1:]=-3.
            self.bin_bias.zero_();self.bin_bias[:,1:]=torch.linspace(-.5,.5,8)[:,None]
            self.distance_effect.fill_(.1)


def fit_sticky_raven(episodes,config,seed=42,steps=80,counterfactual=True):
    model=StickyRegimeValidityAttention(**config);rng=np.random.default_rng(seed)
    opt=torch.optim.Adam(model.parameters(),lr=.015);history=[]
    for step in range(steps):
        e=episodes[int(rng.integers(len(episodes)))];g=e['groups']
        unit=rng.choice(np.unique(g));available=np.flatnonzero(g==unit)
        ix=rng.choice(available,min(12,len(available)),replace=False)
        p=subset_prepared(e['prepared'],ix);y=torch.tensor(e['y'][ix],dtype=torch.float64)
        noise=torch.tensor(rng.normal(size=(8,3)),dtype=torch.float64)
        draws,prob,a,gate=model(p,noise);risk=mixture_energy(draws,prob,y)
        # Weak sticky prior: prevent unsupported transition-everywhere collapse.
        sticky=-torch.log(prob[:,0].clamp_min(1e-8)).mean() if model.transitions else risk.mean()*0
        loss=risk.mean()+.002*gate.mean()+.01*sticky
        if counterfactual and step%4==0:
            importance=[]
            for w in range(4):
                masked=PreparedWindows(p.z,p.theta,p.valid.clone(),p.margin,p.distance,p.base_cov,p.population_cov,p.population_mean)
                can=masked.valid[:,w]&(masked.valid.sum(1)>1);masked.valid[can,w]=False
                cd,cp,_,_=model(masked,noise);importance.append((mixture_energy(cd,cp,y)-risk).detach())
            target=torch.stack(importance,1).softmax(1);observed=a.mean(1).clamp_min(1e-10)
            loss=loss-.05*(target*observed.log()).sum(1).mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite sticky RAVEN objective')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%20==0:history.append(dict(loss=float(loss.detach()),no_switch=float(prob[:,0].mean().detach())))
    return model,dict(seed=seed,steps=steps,counterfactual=counterfactual,sticky_weight=.01,history=history,
        parameters=sum(v.numel() for v in model.parameters()))
