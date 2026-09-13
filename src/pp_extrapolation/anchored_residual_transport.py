"""Zero-start residuals over a frozen direct CIST baseline.

Integral residuals are anchored at normalized progress zero, with bounded signed
rates. They do not guarantee monotonicity of the total prediction. A matched point
residual control separates the integral operator from extra capacity and training.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .model import equal_group_weights
from .innovation_slope_transport import InnovationSlopeTransportFit


class ZeroStartResidual(nn.Module):
    def __init__(self, dimensions, progress_index, route, width=16, steps=4):
        super().__init__()
        if route not in ('integral', 'point'):
            raise ValueError('route must be integral or point')
        self.route=route
        self.index=int(progress_index) % dimensions
        self.steps=int(steps)
        self.net=nn.Sequential(nn.Linear(dimensions,width),nn.ReLU(),
            nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        if self.route=='point':
            return torch.tanh(self.net(x).squeeze(-1))
        t=x[:,self.index]
        midpoint=(torch.arange(self.steps,device=x.device,dtype=x.dtype)+.5)/self.steps
        path=x[:,None,:].expand(-1,self.steps,-1).clone()
        path[:,:,self.index]=t[:,None]*midpoint[None,:]
        rate=torch.tanh(self.net(path).squeeze(-1))
        return -t*rate.mean(1)


@dataclass
class AnchoredResidualFit:
    base: InnovationSlopeTransportFit
    residual: ZeroStartResidual
    cap: float
    accepted: bool
    selected_epoch: int
    diagnostics: dict


def standardized(base,x):
    return torch.as_tensor((np.asarray(x)-base.x_center)/base.x_scale,dtype=torch.float32)


def fit_anchored_residual(base,train,validation,*,seed,route,width=16,steps=4,
                          max_epochs=60,patience=20,penalty=.1):
    base.model.eval()
    base.model.requires_grad_(False)
    x=standardized(base,train['x']);vx=standardized(base,validation['x'])
    y=torch.as_tensor((train['y']-base.y_center)/base.y_scale,dtype=torch.float32)
    vy=np.asarray(validation['y'],dtype=float)
    with torch.no_grad():
        baseline=base.model.mean(x,samples=128)
        val_baseline=base.model.mean(vx,samples=128)
    cap=max(float(np.max(train['y'])),1.)
    torch.manual_seed(seed+3000)
    residual=ZeroStartResidual(x.shape[1],base.progress_index,route,width,steps)
    weights=torch.as_tensor(equal_group_weights(train['groups']),dtype=torch.float32)
    masks=[validation['groups']==g for g in np.unique(validation['groups'])]
    def risk():
        residual.eval()
        with torch.no_grad():
            p=(val_baseline+residual(vx)).numpy()*base.y_scale+base.y_center
        err=(np.clip(p,0,cap)-vy)**2
        return float(err.mean()),float(max(err[m].mean() for m in masks))
    baseline_mse,baseline_worst=risk()
    best=baseline_mse;best_worst=baseline_worst;selected_epoch=0
    best_state=copy.deepcopy(residual.state_dict())
    optimizer=torch.optim.AdamW(residual.parameters(),lr=1e-3,weight_decay=.1)
    rng=np.random.default_rng(seed+4000)
    for epoch in range(1,max_epochs+1):
        residual.train()
        order=rng.permutation(len(x))
        for start in range(0,len(x),512):
            ix=torch.as_tensor(order[start:start+512])
            correction=residual(x[ix])
            error=baseline[ix]+correction-y[ix]
            loss=torch.mean(weights[ix]*(error.abs()+.5*error.square()+penalty*correction.square()))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite residual loss')
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(residual.parameters(),2.)
            optimizer.step()
        mse,worst=risk()
        if mse <= baseline_mse*.99 and worst <= baseline_worst*1.05 and mse < best-1e-10:
            best=mse;best_worst=worst;selected_epoch=epoch
            best_state=copy.deepcopy(residual.state_dict())
        if epoch-selected_epoch>patience:
            break
    residual.load_state_dict(best_state)
    residual.eval()
    return AnchoredResidualFit(base,residual,cap,selected_epoch>0,selected_epoch,
        dict(route=route,baseline_validation_mse=baseline_mse,validation_mse=best,
             baseline_worst_group_mse=baseline_worst,worst_group_mse=best_worst,
             executed_epochs=epoch,penalty=penalty,
             residual_parameters=sum(p.numel() for p in residual.parameters())))


def predict_anchored_residual(fit,x):
    value=standardized(fit.base,x)
    fit.base.model.eval();fit.residual.eval()
    with torch.no_grad():
        p=fit.base.model.mean(value,samples=128)+fit.residual(value)
    return np.clip(p.numpy()*fit.base.y_scale+fit.base.y_center,0,fit.cap)
