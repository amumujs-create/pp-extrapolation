"""Lifetime-scale PP: estimate a unit's latent total life, then subtract age.

This module removes the direct-RUL clipping contract from the original PP.  It
learns log(total life) with a frozen affine path and a bounded nonlinear
correction.  The correction decays outside training support, so extrapolation
falls back continuously to the scale prior.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge


class LifetimeScaleNet(nn.Module):
    def __init__(self, d: int, width: int, residual_bound: float, support_decay: float):
        super().__init__()
        self.affine=nn.Linear(d,1)
        self.residual=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        self.residual_bound=float(residual_bound); self.support_decay=float(support_decay)
        self.register_buffer("support_min",torch.zeros(d)); self.register_buffer("support_max",torch.zeros(d))

    def forward(self,x):
        below=torch.relu(self.support_min-x); above=torch.relu(x-self.support_max)
        distance=torch.linalg.vector_norm(below+above,dim=1,keepdim=True)
        gate=torch.exp(-self.support_decay*distance)
        return (self.affine(x)+gate*self.residual_bound*torch.tanh(self.residual(x))).squeeze(1)


@dataclass
class LifetimeScaleFit:
    model: LifetimeScaleNet
    center: np.ndarray
    scale: np.ndarray
    selection: dict


def _check(split, elapsed):
    x=np.asarray(split["x"],float); y=np.asarray(split["y"],float); g=np.asarray(split["groups"])
    e=np.asarray(elapsed,float)
    if x.ndim!=2 or y.shape!=e.shape!=(len(x),) or g.shape!=(len(x),): raise ValueError("unaligned split")
    if np.any(y<0) or np.any(e<=0) or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("invalid lifetime data")
    return x,y,g,e


def fit_lifetime_scale_pp(train, validation, train_elapsed, validation_elapsed, *, seed=42,
                          width=32, learning_rate=1e-3, weight_decay=.1,
                          residual_bound=1.5, support_decay=.3, max_epochs=350, patience=60):
    tx,ty,tg,te=_check(train,train_elapsed); vx,vy,_,ve=_check(validation,validation_elapsed)
    center=tx.mean(0); scale=tx.std(0); scale[scale<1e-8]=1
    z=((tx-center)/scale).astype("float32"); vz=((vx-center)/scale).astype("float32")
    loglife=np.log(np.maximum(te+ty,1e-6)); vloglife=np.log(np.maximum(ve+vy,1e-6))
    # Affine scale path is selected using validation RUL, not transformed-target error.
    choices=[]
    for alpha in (.01,.1,1.,10.,100.,1000.):
        m=Ridge(alpha=alpha).fit(z,loglife)
        vp=np.maximum(np.exp(np.clip(m.predict(vz),-10,20))-ve,0)
        choices.append((float(np.mean((vp-vy)**2)),alpha,m))
    _,alpha,affine=min(choices,key=lambda q:q[0])
    torch.manual_seed(int(seed)); model=LifetimeScaleNet(z.shape[1],int(width),residual_bound,support_decay)
    with torch.no_grad():
        model.affine.weight.copy_(torch.tensor(affine.coef_[None,:],dtype=torch.float32)); model.affine.bias.copy_(torch.tensor([affine.intercept_],dtype=torch.float32))
        model.support_min.copy_(torch.tensor(z.min(0))); model.support_max.copy_(torch.tensor(z.max(0)))
    model.affine.requires_grad_(False)
    opt=torch.optim.AdamW(model.residual.parameters(),lr=learning_rate,weight_decay=weight_decay)
    xt=torch.tensor(z); yt=torch.tensor(loglife,dtype=torch.float32); et=torch.tensor(te,dtype=torch.float32)
    rawy=torch.tensor(ty,dtype=torch.float32); xval=torch.tensor(vz)
    _,inv,cnt=np.unique(tg,return_inverse=True,return_counts=True); weights=torch.tensor(1/cnt[inv],dtype=torch.float32); weights/=weights.mean()
    rng=np.random.default_rng(seed); best=float("inf"); best_epoch=0; state=copy.deepcopy(model.state_dict()); history=[]
    target_scale=max(float(np.std(ty)),1.)
    for epoch in range(1,max_epochs+1):
        model.train()
        for start in range(0,len(z),512):
            ix=torch.tensor(rng.permutation(len(z))[start:start+512])
            predlog=model(xt[ix]); predlife=torch.exp(torch.clamp(predlog,-10,20)); predrul=torch.relu(predlife-et[ix])
            logloss=torch.nn.functional.smooth_l1_loss(predlog,yt[ix],reduction="none")
            rulloss=((predrul-rawy[ix])/target_scale)**2
            loss=torch.mean(weights[ix]*(logloss+.1*rulloss))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.residual.parameters(),2.); opt.step()
        model.eval()
        with torch.no_grad(): vp=np.maximum(np.exp(np.clip(model(xval).numpy(),-10,20))-ve,0); score=float(np.mean((vp-vy)**2))
        history.append({"epoch":epoch,"validation_rmse":score**.5})
        if score<best-1e-8: best=score; best_epoch=epoch; state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>patience: break
    model.load_state_dict(state)
    return LifetimeScaleFit(model,center.astype("float32"),scale.astype("float32"),{
        "affine_alpha":alpha,"width":width,"learning_rate":learning_rate,"weight_decay":weight_decay,
        "residual_bound":residual_bound,"support_decay":support_decay,"selected_epoch":best_epoch,
        "validation_rmse":best**.5,"history":history})


def predict_lifetime_scale(fit, x, elapsed):
    z=((np.asarray(x,float)-fit.center)/fit.scale).astype("float32")
    fit.model.eval()
    with torch.no_grad(): life=np.exp(np.clip(fit.model(torch.tensor(z)).numpy(),-10,20))
    return np.maximum(life-np.asarray(elapsed,float),0.)
