"""Nested out-of-fold epistemic targets and a deployable uncertainty head."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .model import fit_feature_scale, fit_pp, select_affine_initialization, transform_features
from .support_gate import predict_components

@dataclass
class UncertaintyHeadFit:
    model: nn.Module
    center: np.ndarray
    scale: np.ndarray
    target_scale: float
    oof_target_mean: float
    oof_target_sd: float

class UncertaintyHead(nn.Module):
    def __init__(self,d,width=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1),nn.Softplus())
    def forward(self,x): return self.net(x).squeeze(1)

def _part(split,index): return {k:np.asarray(v)[index] for k,v in split.items()}

def nested_oof_disagreement(train,*,outer_folds=3,teacher_seeds=(101,102,103),max_epochs=200,split_seed=913):
    """Predict each unit only with teachers that never saw that unit or its labels."""
    groups=np.asarray(train['groups']); unique=np.unique(groups)
    if len(unique)<6: raise ValueError('at least six training groups required for nested OOF uncertainty')
    rng=np.random.default_rng(int(split_seed)); unique=unique[rng.permutation(len(unique))]
    fold_ids=np.array_split(unique,int(min(outer_folds,len(unique))))
    target=np.empty(len(groups),dtype=float)
    for outer in fold_ids:
        held=np.isin(groups,outer); remaining=unique[~np.isin(unique,outer)]
        inner_val=remaining[::5]
        inner_train=~np.isin(groups,np.concatenate((outer,inner_val)))
        inner_validation=np.isin(groups,inner_val)
        tr=_part(train,inner_train); va=_part(train,inner_validation)
        affine=select_affine_initialization(tr,va)
        residual=[]
        for seed in teacher_seeds:
            fit=fit_pp(tr,va,seed=int(seed),affine_selection=affine,max_epochs=max_epochs)
            residual.append(predict_components(fit,np.asarray(train['x'])[held])[1]/fit.target_scale)
        target[held]=np.std(np.asarray(residual),axis=0)
    return target

def fit_uncertainty_head(train,oof_target,*,seed=777,max_epochs=500):
    x=np.asarray(train['x']); target=np.asarray(oof_target,dtype=float)
    if target.shape!=(len(x),) or not np.isfinite(target).all() or np.min(target)<0: raise ValueError('invalid OOF target')
    center,scale=fit_feature_scale(x); z=torch.as_tensor(transform_features(x,center,scale),dtype=torch.float32)
    target_scale=max(float(np.quantile(target,.95)),1e-6); y=torch.as_tensor(target/target_scale,dtype=torch.float32)
    torch.manual_seed(seed); model=UncertaintyHead(z.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    best=float('inf');state=copy.deepcopy(model.state_dict());rng=np.random.default_rng(seed)
    for _ in range(max_epochs):
        order=rng.permutation(len(z))
        for start in range(0,len(z),512):
            ix=torch.as_tensor(order[start:start+512]); loss=torch.mean((model(z[ix])-y[ix])**2)
            opt.zero_grad();loss.backward();opt.step()
        value=float(torch.mean((model(z)-y)**2).detach())
        if value<best: best=value;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return UncertaintyHeadFit(model,center,scale,target_scale,float(target.mean()),float(target.std()))

def predict_uncertainty(head,x):
    z=torch.as_tensor(transform_features(x,head.center,head.scale),dtype=torch.float32);head.model.eval()
    with torch.no_grad(): value=head.model(z).numpy()*head.target_scale
    return np.maximum(value,0.)
