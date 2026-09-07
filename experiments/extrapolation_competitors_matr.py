#!/usr/bin/env python3
"""External extrapolation-oriented competitors on the locked MATR split.

All hyperparameters are chosen on validation cells. Test labels are materialized
only after selection. Cell ids are the environments for V-REx and GroupDRO.
"""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'.benchmark_deps')]
from extended_nn_benchmark import data
from pp_extrapolation import regression_metrics,select_affine_initialization
from pp_extrapolation.model import transform_features,equal_group_weights

OUT=ROOT/'results/extrapolation_competitors_matr_v1'
SEEDS=range(42,47)
ARCH=[(32,2,5e-4,.1),(64,2,5e-4,.1)]
PENALTIES=(.01,.1,1.)

class MLP(nn.Module):
 def __init__(self,d,w,depth):
  super().__init__()
  layers=[]
  for i in range(depth):layers += [nn.Linear(d if i==0 else w,w),nn.Tanh()]
  layers += [nn.Linear(w,1)];self.net=nn.Sequential(*layers)
 def forward(self,x):return self.net(x).squeeze(-1)

def group_risks(error,groups):
 return torch.stack([error[groups==u].mean() for u in torch.unique(groups)])

def fit_neural(kind,cfg,seed,parts,aff,return_test=False):
 tr,va,te=parts;w,depth,lr,wd,lam=cfg;cap=float(aff['target_scale'])
 arrays=[torch.tensor(transform_features(p['x'],aff['center'],aff['scale']),dtype=torch.float32) for p in parts]
 x,v,t=arrays;y=torch.tensor(tr['y']/cap,dtype=torch.float32);vy=torch.tensor(va['y']/cap,dtype=torch.float32)
 labels,inv=np.unique(tr['groups'],return_inverse=True);gi=torch.tensor(inv);weights=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32)
 mono_sign=1. if np.corrcoef(tr['x'][:,0],tr['y'])[0,1]>=0 else -1.
 torch.manual_seed(seed);model=MLP(x.shape[1],w,depth);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
 q=torch.ones(len(labels))/len(labels);best=np.inf;state=copy.deepcopy(model.state_dict());be=0;rng=np.random.default_rng(seed);started=time.monotonic()
 for epoch in range(1,201):
  model.train();ix=rng.permutation(len(x))
  for start in range(0,len(x),1024):
   b=torch.tensor(ix[start:start+1024]);xb=x[b]
   if kind=='monotone':xb=xb.detach().requires_grad_(True)
   err=(model(xb)-y[b]).square()
   if kind=='vrex':
    risks=group_risks(err,gi[b]);loss=risks.mean()+lam*risks.var(unbiased=False)
   elif kind=='groupdro':
    risks=group_risks(err,gi[b]);present=torch.unique(gi[b]);
    with torch.no_grad():q[present]*=torch.exp(lam*risks.detach());q/=q.sum()
    loss=(q[present]/q[present].sum()*risks).sum()
   else:
    pred=model(xb);grad=torch.autograd.grad(pred.sum(),xb,create_graph=True)[0][:,0]
    loss=(weights[b]*(pred-y[b]).square()).mean()+lam*torch.relu(-mono_sign*grad).square().mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  model.eval()
  with torch.no_grad():score=float(((torch.clamp(model(v),0,1)-vy)**2).mean())
  if score<best-1e-10:best=score;be=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-be>=30:break
 model.load_state_dict(state);info={'validation_mse':best*cap*cap,'selected_epoch':be,'epochs':epoch,'seconds':time.monotonic()-started}
 if not return_test:return info,None
 model.eval()
 with torch.no_grad():p=np.clip(model(t).numpy()*cap,0,cap)
 return info,p

def neural_model(kind,parts,aff,save_prefix=''):
 grid=[(*a,lam) for a in ARCH for lam in PENALTIES];search=[]
 for cfg in grid:
  info,_=fit_neural(kind,cfg,42,parts,aff);search.append({'config':cfg,**info});print('SEARCH',kind,cfg,info['validation_mse'],flush=True)
 chosen=grid[int(np.argmin([r['validation_mse'] for r in search]))];runs=[];pred=[]
 for seed in SEEDS:
  info,p=fit_neural(kind,chosen,seed,parts,aff,True);pred.append(p);runs.append({'seed':seed,**info,'metrics':regression_metrics(parts[2]['y'],p,parts[2]['groups'])});print('REFIT',kind,seed,flush=True)
 ens=regression_metrics(parts[2]['y'],np.mean(pred,axis=0),parts[2]['groups'])
 np.savez_compressed(OUT/f'{save_prefix}{kind}_predictions.npz',prediction=np.asarray(pred),y=parts[2]['y'],groups=parts[2]['groups'])
 return {'protocol':'validation-only selection; five refits','selected_config':chosen,'search':search,'runs':runs,'ensemble':ens,'mean_r2':float(np.mean([r['metrics']['pooled']['r2'] for r in runs])),'sd_r2':float(np.std([r['metrics']['pooled']['r2'] for r in runs],ddof=1))}

def linear_rff(parts,aff,save_prefix=''):
 tr,va,te=parts;zs=[transform_features(p['x'],aff['center'],aff['scale']) for p in parts];cap=float(aff['target_scale']);grid=[(g,a) for g in (.03,.1,.3,1.) for a in (.1,1.,10.)];search=[]
 def design(z,rff):return np.column_stack([np.ones(len(z)),z,rff.transform(z)])
 for gamma,alpha in grid:
  rff=RBFSampler(gamma=gamma,n_components=384,random_state=42).fit(zs[0]);m=Ridge(alpha=alpha,fit_intercept=False).fit(design(zs[0],rff),tr['y']);p=np.clip(m.predict(design(zs[1],rff)),0,cap);search.append({'gamma':gamma,'alpha':alpha,'validation_mse':float(np.mean((p-va['y'])**2))})
 chosen=min(search,key=lambda r:r['validation_mse']);pred=[];runs=[]
 for seed in SEEDS:
  rff=RBFSampler(gamma=chosen['gamma'],n_components=384,random_state=seed).fit(zs[0]);m=Ridge(alpha=chosen['alpha'],fit_intercept=False).fit(design(zs[0],rff),tr['y']);p=np.clip(m.predict(design(zs[2],rff)),0,cap);pred.append(p);runs.append({'seed':seed,'metrics':regression_metrics(te['y'],p,te['groups'])})
 ens=regression_metrics(te['y'],np.mean(pred,axis=0),te['groups'])
 np.savez_compressed(OUT/f'{save_prefix}linear_rff_predictions.npz',prediction=np.asarray(pred),y=te['y'],groups=te['groups'])
 return {'protocol':'linear extrapolating mean plus random Fourier RBF residual; validation-only selection','selected_config':chosen,'search':search,'runs':runs,'ensemble':ens,'mean_r2':float(np.mean([r['metrics']['pooled']['r2'] for r in runs])),'sd_r2':float(np.std([r['metrics']['pooled']['r2'] for r in runs],ddof=1))}

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);parts=data();aff=select_affine_initialization(parts[0],parts[1]);result={'split':'locked MATR2019 train/validation/test cells and strict health-tail rows','models':{}}
 for kind in ('vrex','groupdro','monotone'):result['models'][kind]=neural_model(kind,parts,aff);(OUT/'results.partial.json').write_text(json.dumps(result,indent=2)+'\n')
 result['models']['linear_rff']=linear_rff(parts,aff);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 for k,v in result['models'].items():print('DONE',k,v['ensemble']['pooled']['r2'],flush=True)
if __name__=='__main__':main()
