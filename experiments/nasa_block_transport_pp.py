#!/usr/bin/env python3
"""Blocked-validation transport on the fully tuned NASA PP."""
import json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from run_affine_tail_external_nasa_health_v2 import prepare_folds
OUT=ROOT/'results/nasa_block_transport_pp_v1';SEEDS=range(42,47);ALPHAS=(.1,1.,10.,100.,1000.)
def design(kind,p,x,mu=None,sd=None):
 if kind=='affine':return np.c_[p]
 z=x
 if mu is None:mu=z.mean(0);sd=np.maximum(z.std(0),1e-8)
 return np.c_[p,(z-mu)/sd],mu,sd
def cv(kind,alpha,y,p,x,cap):
 blocks=np.array_split(np.arange(len(y)),4);err=[]
 for b in blocks:
  keep=np.ones(len(y),bool);keep[b]=False
  if kind=='affine':A=design(kind,p[keep],x[keep]);B=design(kind,p[b],x[b])
  else:A,mu,sd=design(kind,p[keep],x[keep]);B,_,_=design(kind,p[b],x[b],mu,sd)
  q=np.clip(Ridge(alpha=alpha).fit(A,y[keep]).predict(B),0,cap);err.extend((q-y[b])**2)
 return np.mean(err)
def apply(kind,alpha,y,p,x,tp,tx,cap):
 if kind=='affine':A=design(kind,p,x);B=design(kind,tp,tx)
 else:A,mu,sd=design(kind,p,x);B,_,_=design(kind,tp,tx,mu,sd)
 return np.clip(Ridge(alpha=alpha).fit(A,y).predict(B),0,cap)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);tune=json.load(open(ROOT/'results/nasa_pp_full_tuning_v1/results.json'));folds,_=prepare_folds();pred=[[] for _ in SEEDS];ys=[];gs=[];audit=[]
 for fi,f in enumerate(folds):
  tr,va,te=f['train'],f['validation'],f['test'];c=tune['folds'][fi]['selected'];a=select_affine_initialization(tr,va);choices=[]
  for si,s in enumerate(SEEDS):
   m=fit_pp(tr,va,seed=s,affine_selection=a,max_epochs=400,patience=80,width=c['width'],learning_rate=c['lr'],weight_decay=c['wd'],group_dro_eta=c['eta'],residual_decay=c['decay'],affine_anchor_weight=c['anchor']);vp=predict(m,va['x']);tp=predict(m,te['x']);cand=[{'kind':'identity','alpha':0.,'mse':float(np.mean((vp-va['y'])**2))}]
   for k in ('affine','state'):
    for al in ALPHAS:cand.append({'kind':k,'alpha':al,'mse':float(cv(k,al,va['y'],vp,va['x'],m.target_scale))})
   z=min(cand,key=lambda q:q['mse']);q=tp if z['kind']=='identity' else apply(z['kind'],z['alpha'],va['y'],vp,va['x'],tp,te['x'],m.target_scale);pred[si].append(q);choices.append({'seed':s,'chosen':z})
  ys.append(te['y']);gs.append(te['groups']);audit.append({'test_cell':f['test_cell'],'choices':choices})
 y=np.concatenate(ys);g=np.concatenate(gs);matrix=np.asarray([np.concatenate(p) for p in pred]);r={'selection':'4 contiguous validation blocks only','folds':audit,'ensemble':regression_metrics(y,matrix.mean(0),g)};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
