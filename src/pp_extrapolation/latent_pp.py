"""PP for signed latent coordinates with an arbitrary differentiable decoder."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge

class LatentPPNet(nn.Module):
    def __init__(self,d,width,bound,decay):
        super().__init__(); self.affine=nn.Linear(d,1); self.bound=float(bound); self.decay=float(decay)
        self.nn=nn.Sequential(nn.Linear(d,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.nn[-1].weight); nn.init.zeros_(self.nn[-1].bias)
        self.register_buffer("lo",torch.zeros(d)); self.register_buffer("hi",torch.zeros(d))
    def forward(self,x):
        distance=torch.linalg.vector_norm(torch.relu(self.lo-x)+torch.relu(x-self.hi),dim=1,keepdim=True)
        return (self.affine(x)+torch.exp(-self.decay*distance)*self.bound*torch.tanh(self.nn(x))).squeeze(1)

@dataclass
class LatentFit:
    model: LatentPPNet; center: np.ndarray; scale: np.ndarray; selection: dict

def fit_latent_pp(train,validation,train_target,validation_target,validation_decoder,*,seed=42,width=32,
                  learning_rate=1e-3,weight_decay=.1,residual_bound=1.,support_decay=.3,max_epochs=300,patience=50):
    tx=np.asarray(train["x"],float); vx=np.asarray(validation["x"],float); target=np.asarray(train_target,float); vt=np.asarray(validation_target,float)
    center=tx.mean(0); scale=tx.std(0); scale[scale<1e-8]=1; z=((tx-center)/scale).astype("float32"); vz=((vx-center)/scale).astype("float32")
    candidates=[]
    for alpha in (.01,.1,1,10,100,1000):
        m=Ridge(alpha=alpha).fit(z,target); score=float(np.mean((validation_decoder(m.predict(vz))-validation["y"])**2)); candidates.append((score,alpha,m))
    _,alpha,base=min(candidates,key=lambda q:q[0]); torch.manual_seed(seed); model=LatentPPNet(z.shape[1],width,residual_bound,support_decay)
    with torch.no_grad():
        model.affine.weight.copy_(torch.tensor(base.coef_[None,:],dtype=torch.float32)); model.affine.bias.copy_(torch.tensor([base.intercept_],dtype=torch.float32)); model.lo.copy_(torch.tensor(z.min(0))); model.hi.copy_(torch.tensor(z.max(0)))
    model.affine.requires_grad_(False); opt=torch.optim.AdamW(model.nn.parameters(),lr=learning_rate,weight_decay=weight_decay)
    xt=torch.tensor(z); yt=torch.tensor(target,dtype=torch.float32); xval=torch.tensor(vz); rng=np.random.default_rng(seed)
    best=float("inf"); epoch_best=0; state=copy.deepcopy(model.state_dict())
    for epoch in range(1,max_epochs+1):
        model.train(); order=rng.permutation(len(z))
        for start in range(0,len(z),512):
            ix=torch.tensor(order[start:start+512]); loss=torch.nn.functional.smooth_l1_loss(model(xt[ix]),yt[ix])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.nn.parameters(),2); opt.step()
        model.eval()
        with torch.no_grad(): score=float(np.mean((validation_decoder(model(xval).numpy())-validation["y"])**2))
        if score<best-1e-8: best=score; epoch_best=epoch; state=copy.deepcopy(model.state_dict())
        if epoch-epoch_best>patience: break
    model.load_state_dict(state)
    return LatentFit(model,center.astype("float32"),scale.astype("float32"),{"alpha":alpha,"validation_rmse":best**.5,"epoch":epoch_best})

def predict_latent(fit,x):
    z=((np.asarray(x,float)-fit.center)/fit.scale).astype("float32"); fit.model.eval()
    with torch.no_grad(): return fit.model(torch.tensor(z)).numpy().astype(float)
