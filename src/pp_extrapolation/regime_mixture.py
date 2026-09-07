"""Latent transition mixture for PP deep-future extrapolation."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .model import _arrays,equal_group_weights,select_affine_initialization,transform_features

class LatentRegimePPNet(nn.Module):
    """Frozen affine path with a monotone transition gate and two neural tails."""
    def __init__(self,d,direction,knot,width=24):
        super().__init__();self.direction=float(direction);self.knot=float(knot)
        self.affine=nn.Linear(d,1)
        self.gate_q=nn.Parameter(torch.tensor(-2.));self.gate_bias=nn.Parameter(torch.tensor(-2.))
        self.gate_context=nn.Sequential(nn.Linear(max(d-1,1),width),nn.Tanh(),nn.Linear(width,1))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,width),nn.Tanh(),nn.Linear(width,2)) for _ in range(2)])
        for net in [self.gate_context,*self.experts]:nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    def components(self,x):
        q=self.direction*x[:,0:1];context=x[:,1:] if x.shape[1]>1 else torch.zeros_like(q)
        gate=torch.sigmoid(torch.nn.functional.softplus(self.gate_q)*q+self.gate_bias+self.gate_context(context))
        hinge=torch.relu(q-self.knot);tails=[]
        for expert in self.experts:
            raw=expert(x);tails.append(.25*torch.tanh(raw[:,0:1])+.25*torch.tanh(raw[:,1:2])*hinge)
        correction=(1-gate)*tails[0]+gate*tails[1]
        return self.affine(x)+correction,gate,tails
    def forward(self,x):return self.components(x)[0].squeeze(1)

@dataclass
class LatentRegimeFit:
    model:LatentRegimePPNet;center:np.ndarray;scale:np.ndarray;target_scale:float;selection:dict

def fit_latent_regime_pp(train,validation,*,seed,affine_selection=None,max_epochs=300,patience=70,separation_weight=.01,gate_weight=0.,trainable_affine=False,width=24,learning_rate=5e-4,weight_decay=2.,group_dro_eta=0.,affine_anchor_weight=0.):
    tx,ty,groups=_arrays(train);vx,vy,_=_arrays(validation);sel=affine_selection or select_affine_initialization(train,validation)
    center,scale,cap,init=sel['center'],sel['scale'],float(sel['target_scale']),sel['initialization'];z=transform_features(tx,center,scale);vz=transform_features(vx,center,scale)
    direction=1. if np.mean(vz[:,0])>np.mean(z[:,0]) else -1.;knot=float(np.quantile(direction*z[:,0],.8));torch.manual_seed(seed)
    model=LatentRegimePPNet(z.shape[1],direction,knot,width=width)
    with torch.no_grad():model.affine.weight.copy_(torch.tensor(init.weight)[None,:]);model.affine.bias.copy_(torch.tensor([init.bias]))
    model.affine.requires_grad_(trainable_affine);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=weight_decay)
    x=torch.tensor(z);y=torch.tensor(ty/cap,dtype=torch.float32);w=torch.tensor(equal_group_weights(groups),dtype=torch.float32);val=torch.tensor(vz);rng=np.random.default_rng(seed)
    _,group_index=np.unique(groups,return_inverse=True);group_index=torch.tensor(group_index);dro_q=torch.ones(int(group_index.max())+1);dro_q/=dro_q.sum()
    anchor_w=torch.tensor(init.weight);anchor_b=torch.tensor(init.bias)
    def vl():
        model.eval()
        with torch.no_grad():p=np.clip(model(val).numpy()*cap,0,cap)
        return float(np.mean((p-vy)**2))
    best=vl();be=0;state=copy.deepcopy(model.state_dict())
    for epoch in range(1,max_epochs+1):
        model.train();order=rng.permutation(len(x))
        for start in range(0,len(x),512):
            ix=torch.tensor(order[start:start+512]);pred,gate,tails=model.components(x[ix]);errors=(pred.squeeze(1)-y[ix])**2
            if group_dro_eta>0:
                present=torch.unique(group_index[ix]);risks=torch.stack([errors[group_index[ix]==u].mean() for u in present])
                with torch.no_grad():dro_q[present]*=torch.exp(float(group_dro_eta)*risks.detach());dro_q/=dro_q.sum()
                loss=torch.sum(dro_q[present]/dro_q[present].sum()*risks)
            else:loss=torch.mean(w[ix]*errors)
            # Prevent identical experts without prescribing either regime's equation.
            separation=torch.mean((tails[0]-tails[1])**2);loss=loss-float(separation_weight)*torch.clamp(separation,max=.05)
            # A usable latent partition needs both regimes and decisive assignments.
            gate_regularizer=(gate.mean()-.5).square()+torch.mean(gate*(1.-gate))
            loss=loss+float(gate_weight)*gate_regularizer
            if trainable_affine and affine_anchor_weight>0:
                loss=loss+float(affine_anchor_weight)*(torch.mean((model.affine.weight.squeeze()-anchor_w)**2)+(model.affine.bias.squeeze()-anchor_b)**2)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        cur=vl()
        if cur<best-1e-10:best=cur;be=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-be>patience:break
    model.load_state_dict(state)
    return LatentRegimeFit(model,center,scale,cap,{'seed':seed,'selected_epoch':be,'validation_mse':best,'direction':direction,'knot':knot,'separation_weight':float(separation_weight),'gate_weight':float(gate_weight),'group_dro_eta':float(group_dro_eta),'trainable_affine':bool(trainable_affine),'affine_anchor_weight':float(affine_anchor_weight)})

def predict_latent_regime(fit,x,return_gate=False):
    fit.model.eval();z=torch.tensor(transform_features(x,fit.center,fit.scale))
    with torch.no_grad():pred,gate,_=fit.model.components(z);p=np.clip(pred.squeeze(1).numpy()*fit.target_scale,0,fit.target_scale);g=gate.squeeze(1).numpy()
    return (p,g) if return_gate else p

def latent_regime_components(fit,x):
    """Return the latent PP affine path and neural correction in target units."""
    fit.model.eval();z=torch.tensor(transform_features(x,fit.center,fit.scale))
    with torch.no_grad():
        _,gate,tails=fit.model.components(z)
        affine=fit.model.affine(z).squeeze(1).numpy()*fit.target_scale
        correction=((1-gate)*tails[0]+gate*tails[1]).squeeze(1).numpy()*fit.target_scale
    return affine.astype(np.float64),correction.astype(np.float64)
