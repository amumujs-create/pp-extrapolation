#!/usr/bin/env python3
"""Temporal PP on log1p(remaining-life / observed-position)."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from pp_extrapolation import regression_metrics
from pp_extrapolation.model import equal_group_weights
from xjtu_spectrum_adapter_v2 import load
from xjtu_uncapped_temporal_pp_v2 import rows
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/xjtu_progress_temporal_pp_v3";SEEDS=(42,43,44,45,46)

class Net(nn.Module):
 def __init__(self,d,mode,bound=.5):
  super().__init__();self.mode=mode;self.bound=bound;self.gru=nn.GRU(d,16,batch_first=True);self.direct=nn.Linear(16,1);self.affine=nn.Linear(d,1);self.residual=nn.Linear(16,1);nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
 def forward(self,x):
  _,h=self.gru(x);h=h[-1];raw=self.direct(h).squeeze(1) if self.mode=="direct" else self.affine(x[:,-1]).squeeze(1)+self.bound*torch.tanh(self.residual(h).squeeze(1));return torch.nn.functional.softplus(raw)

def fit(tr,va,seed,mode,bound=.5):
 center=tr["x"][:,-1].mean(0);scale=tr["x"][:,-1].std(0);scale=np.where(scale<1e-8,1,scale)
 tx=torch.tensor((tr["x"]-center)/scale);vx=torch.tensor((va["x"]-center)/scale);target=np.log1p(tr["y"]/tr["positions"]);yt=torch.tensor(target,dtype=torch.float32);w=torch.tensor(equal_group_weights(tr["groups"]),dtype=torch.float32)
 torch.manual_seed(seed);model=Net(tx.shape[-1],mode,bound)
 if mode=="pp":
  candidates=[]
  for alpha in (.1,1,10,100,1000):
   ridge=Ridge(alpha=alpha).fit(tx[:,-1].numpy(),target);latent=np.log1p(np.exp((vx[:,-1].numpy()@ridge.coef_)+ridge.intercept_));p=va["positions"]*np.expm1(np.clip(latent,0,8));candidates.append((np.mean((p-va["y"])**2),ridge))
  ridge=min(candidates,key=lambda q:q[0])[1]
  with torch.no_grad():model.affine.weight.copy_(torch.tensor(ridge.coef_[None],dtype=torch.float32));model.affine.bias.copy_(torch.tensor([ridge.intercept_],dtype=torch.float32))
  model.affine.requires_grad_(False)
 params=list(model.gru.parameters())+list(model.direct.parameters() if mode=="direct" else model.residual.parameters());opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=.05);rng=np.random.default_rng(seed)
 def score():
  model.eval()
  with torch.no_grad():z=model(vx).numpy();p=va["positions"]*np.expm1(np.clip(z,0,8))
  return float(np.mean((p-va["y"])**2))
 best=score();state=copy.deepcopy(model.state_dict());be=0
 for epoch in range(1,261):
  model.train();order=rng.permutation(len(tx))
  for start in range(0,len(tx),512):
   ix=order[start:start+512];loss=torch.mean(w[ix]*(model(tx[ix])-yt[ix]).square());opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,2);opt.step()
  s=score()
  if s<best-1e-8:best=s;be=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-be>50:break
 model.load_state_dict(state);return model,center,scale,{"epoch":be,"validation_rmse":best**.5}
def predict(f,x):
 model,center,scale,_=f;model.eval()
 with torch.no_grad():z=model(torch.tensor((x["x"]-center)/scale)).numpy()
 return x["positions"]*np.expm1(np.clip(z,0,8))
def main():
 torch.set_num_threads(2);raw=load();result={}
 for representation in ("summary","spectrum"):
  tr=rows(raw,"37.5Hz11kN",representation);va=rows(raw,"35Hz12kN",representation);te=rows(raw,"40Hz10kN",representation)
  for mode in ("direct","pp"):
   pred=[];runs=[]
   for seed in SEEDS:
    f=fit(tr,va,seed,mode);p=predict(f,te);pred.append(p);runs.append({"seed":seed,"selection":f[3],"metrics":regression_metrics(te["y"],p,te["groups"])});print(representation,mode,seed,runs[-1]["metrics"]["pooled"]["r2"],flush=True)
   pred=np.asarray(pred);key=f"{representation}_{mode}";result[key]={"runs":runs,"ensemble":regression_metrics(te["y"],pred.mean(0),te["groups"]),"seed_mean":float(np.mean([r["metrics"]["pooled"]["r2"] for r in runs])),"seed_sd":float(np.std([r["metrics"]["pooled"]["r2"] for r in runs],ddof=1))};OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/f"{key}.npz",y=te["y"],groups=te["groups"],predictions=pred)
 (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:v["ensemble"]["pooled"] for k,v in result.items()},indent=2))
if __name__=="__main__":main()
