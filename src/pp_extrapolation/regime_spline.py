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

def fit_regime_spline_pp(train,validation,*,seed,affine_selection=None,max_epochs=300,patience=70,monotone=False,jacobian_weight=0.,jacobian_ray_multiplier=0.):
    tx,ty,groups=_arrays(train);vx,vy,_=_arrays(validation);sel=affine_selection or select_affine_initialization(train,validation);center=sel['center'];scale=sel['scale'];cap=float(sel['target_scale']);init=sel['initialization']
    if not np.isfinite(jacobian_weight) or jacobian_weight<0:raise ValueError('jacobian_weight must be finite and nonnegative')
    if not np.isfinite(jacobian_ray_multiplier) or jacobian_ray_multiplier<0:raise ValueError('jacobian_ray_multiplier must be finite and nonnegative')
    z=transform_features(tx,center,scale);vz=transform_features(vx,center,scale);direction=1. if np.mean(vz[:,0])>np.mean(z[:,0]) else -1.;q=direction*z[:,0];knots=np.quantile(q,[.5,.7,.85,.95]).astype(np.float32)
    torch.manual_seed(seed);model=RegimeSplineNet(z.shape[1],knots,direction,monotone=monotone)
    with torch.no_grad():model.affine.weight.copy_(torch.tensor(init.weight)[None,:]);model.affine.bias.copy_(torch.tensor([init.bias]))
    model.affine.requires_grad_(False);parameters=list(model.local.parameters())+([model.monotone_logits] if monotone else list(model.context.parameters()));opt=torch.optim.AdamW(parameters,lr=5e-4,weight_decay=2.)
    x=torch.tensor(z);y=torch.tensor(ty/cap,dtype=torch.float32);w=torch.tensor(equal_group_weights(groups),dtype=torch.float32);val=torch.tensor(vz);rng=np.random.default_rng(seed)
    ray=None
    if jacobian_weight>0 and jacobian_ray_multiplier>0:
        # Reuse observed contexts, but advance only q beyond labelled training support.
        # Its horizon is fixed from the train-validation gap and never sees test labels/features.
        n=min(2048,len(z));ray=z[rng.choice(len(z),n,replace=len(z)<n)].copy()
        q_train_max=float(np.max(q));q_val_max=float(np.max(direction*vz[:,0]));gap=max(q_val_max-q_train_max,1e-3)
        q_future=rng.uniform(q_train_max,q_val_max+jacobian_ray_multiplier*gap,n)
        ray[:,0]=direction*q_future;ray=torch.tensor(ray,dtype=torch.float32)
    def vl():
        model.eval()
        with torch.no_grad():p=np.clip(model(val).numpy()*cap,0,cap)
        return float(np.mean((p-vy)**2))
    best=vl();be=0;state=copy.deepcopy(model.state_dict())
    for epoch in range(1,max_epochs+1):
        model.train();order=rng.permutation(len(x))
        for start in range(0,len(x),512):
            ix=torch.tensor(order[start:start+512]);xb=x[ix].detach().requires_grad_(jacobian_weight>0);prediction=model(xb);loss=torch.mean(w[ix]*(prediction-y[ix])**2)
            if jacobian_weight>0:
                penalty_x=xb
                if ray is not None:
                    ri=torch.tensor(rng.integers(0,len(ray),size=min(512,len(ray))));penalty_x=ray[ri].detach().requires_grad_(True)
                    penalty_prediction=model(penalty_x)
                else: penalty_prediction=prediction
                gradient=torch.autograd.grad(penalty_prediction.sum(),penalty_x,create_graph=True)[0]
                # q=direction*z0. Positive dy/dq means RUL increases as degradation advances.
                violation=torch.relu(direction*gradient[:,0]).square().mean()
                loss=loss+float(jacobian_weight)*violation
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        cur=vl()
        if cur<best-1e-10:best=cur;be=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-be>patience:break
    model.load_state_dict(state);return RegimeSplineFit(model,center,scale,cap,{'seed':seed,'selected_epoch':be,'validation_mse':best,'direction':direction,'knots':knots.tolist(),'monotone':bool(monotone),'jacobian_weight':float(jacobian_weight),'jacobian_ray_multiplier':float(jacobian_ray_multiplier)})

def predict_regime_spline(fit,x):
    fit.model.eval()
    with torch.no_grad():p=fit.model(torch.tensor(transform_features(x,fit.center,fit.scale))).numpy()*fit.target_scale
    return np.clip(p,0,fit.target_scale)

def jacobian_diagnostics(fit,x):
    """Return total-output monotonicity violations along oriented degradation."""
    fit.model.eval();z=torch.tensor(transform_features(x,fit.center,fit.scale),requires_grad=True)
    prediction=fit.model(z);gradient=torch.autograd.grad(prediction.sum(),z)[0]
    oriented=(float(fit.selection['direction'])*gradient[:,0]).detach().numpy()
    positive=np.maximum(oriented,0.)
    return {'violation_fraction':float(np.mean(oriented>0)),
            'mean_positive_derivative':float(np.mean(positive)),
            'max_positive_derivative':float(np.max(positive))}
