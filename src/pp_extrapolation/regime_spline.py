"""Regime-adaptive PP with a neural contextual linear-spline tail."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .model import (_arrays,equal_group_weights,select_affine_initialization,
                    transform_features)

class RegimeSplineNet(nn.Module):
    """Frozen affine path plus context-conditioned hinge slopes and local residual."""
    def __init__(self,d,knots,direction,width=24,monotone=False):
        super().__init__();self.affine=nn.Linear(d,1);self.register_buffer('knots',torch.tensor(knots,dtype=torch.float32));self.direction=float(direction)
        self.monotone=bool(monotone);self.context=nn.Sequential(nn.Linear(d,width),nn.Tanh(),nn.Linear(width,len(knots)))
        self.monotone_logits=nn.Parameter(torch.full((len(knots),),-6.))
        self.local=nn.Sequential(nn.Linear(d,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias);nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
    def forward(self,x):
        q=self.direction*x[:,0:1];hinge=torch.relu(q-self.knots[None,:]);slopes=(-.25*torch.nn.functional.softplus(self.monotone_logits)[None,:] if self.monotone else .25*torch.tanh(self.context(x)));spline=(slopes*hinge).sum(1,keepdim=True)
        local=.25*torch.tanh(self.local(x));return (self.affine(x)+spline+local).squeeze(1)

@dataclass
class RegimeSplineFit:
    model:RegimeSplineNet;center:np.ndarray;scale:np.ndarray;target_scale:float;selection:dict

def fit_regime_spline_pp(train,validation,*,seed,affine_selection=None,max_epochs=300,patience=70,monotone=False):
    tx,ty,groups=_arrays(train);vx,vy,_=_arrays(validation);sel=affine_selection or select_affine_initialization(train,validation);center=sel['center'];scale=sel['scale'];cap=float(sel['target_scale']);init=sel['initialization']
    z=transform_features(tx,center,scale);vz=transform_features(vx,center,scale);direction=1. if np.mean(vz[:,0])>np.mean(z[:,0]) else -1.;q=direction*z[:,0];knots=np.quantile(q,[.5,.7,.85,.95]).astype(np.float32)
    torch.manual_seed(seed);model=RegimeSplineNet(z.shape[1],knots,direction,monotone=monotone)
    with torch.no_grad():model.affine.weight.copy_(torch.tensor(init.weight)[None,:]);model.affine.bias.copy_(torch.tensor([init.bias]))
    model.affine.requires_grad_(False);parameters=list(model.local.parameters())+([model.monotone_logits] if monotone else list(model.context.parameters()));opt=torch.optim.AdamW(parameters,lr=5e-4,weight_decay=2.)
    x=torch.tensor(z);y=torch.tensor(ty/cap,dtype=torch.float32);w=torch.tensor(equal_group_weights(groups),dtype=torch.float32);val=torch.tensor(vz);rng=np.random.default_rng(seed)
    def vl():
        model.eval()
        with torch.no_grad():p=np.clip(model(val).numpy()*cap,0,cap)
        return float(np.mean((p-vy)**2))
    best=vl();be=0;state=copy.deepcopy(model.state_dict())
    for epoch in range(1,max_epochs+1):
        model.train();order=rng.permutation(len(x))
        for start in range(0,len(x),512):
            ix=torch.tensor(order[start:start+512]);loss=torch.mean(w[ix]*(model(x[ix])-y[ix])**2)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        cur=vl()
        if cur<best-1e-10:best=cur;be=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-be>patience:break
    model.load_state_dict(state);return RegimeSplineFit(model,center,scale,cap,{'seed':seed,'selected_epoch':be,'validation_mse':best,'direction':direction,'knots':knots.tolist(),'monotone':bool(monotone)})

def predict_regime_spline(fit,x):
    fit.model.eval()
    with torch.no_grad():p=fit.model(torch.tensor(transform_features(x,fit.center,fit.scale))).numpy()*fit.target_scale
    return np.clip(p,0,fit.target_scale)
