#!/usr/bin/env python3
"""Validation-group-LOO regime-conditioned output transport for HUST PP."""
import json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from run_affine_tail_external_three import prepare_hust
OUT=ROOT/'results/hust_regime_transport_pp_v1';SEEDS=range(42,47);ALPHAS=(.1,1.,10.,100.,1000.)

def design(kind,p,x,mu=None,sd=None):
 if kind=='affine':return np.c_[p]
 cols={'state':(0,1),'rate':(2,3),'all':(0,1,2,3)}[kind];z=x[:,cols]
 if mu is None:mu=z.mean(0);sd=np.maximum(z.std(0),1e-8)
 return np.c_[p,(z-mu)/sd],mu,sd

def loo(kind,alpha,y,p,x,g,cap):
 errors=[]
 for u in np.unique(g):
  a=g!=u
  if kind=='affine':A=design(kind,p[a],x[a]);B=design(kind,p[~a],x[~a])
  else:A,mu,sd=design(kind,p[a],x[a]);B,_,_=design(kind,p[~a],x[~a],mu,sd)
  m=Ridge(alpha=alpha).fit(A,y[a]);q=np.clip(m.predict(B),0,cap);errors.extend((q-y[~a])**2)
 return float(np.mean(errors))

def calibrate(kind,alpha,y,p,x,test_p,test_x,cap):
 if kind=='affine':A=design(kind,p,x);B=design(kind,test_p,test_x)
 else:A,mu,sd=design(kind,p,x);B,_,_=design(kind,test_p,test_x,mu,sd)
 return np.clip(Ridge(alpha=alpha).fit(A,y).predict(B),0,cap)

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);tr,va,te=prepare_hust()[:3];aff=select_affine_initialization(tr,va);raw=[];final=[];runs=[]
 for seed in SEEDS:
  f=fit_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=300,patience=70,width=64,affine_anchor_weight=.1);vp=predict(f,va['x']);tp=predict(f,te['x']);candidates=[]
  for kind in ('affine','state','rate','all'):
   for alpha in ALPHAS:candidates.append({'kind':kind,'alpha':alpha,'loo_mse':loo(kind,alpha,va['y'],vp,va['x'],va['groups'],f.target_scale)})
  chosen=min(candidates,key=lambda z:z['loo_mse']);q=calibrate(chosen['kind'],chosen['alpha'],va['y'],vp,va['x'],tp,te['x'],f.target_scale);raw.append(tp);final.append(q);runs.append({'seed':seed,'chosen':chosen,'candidates':candidates});print(seed,chosen,flush=True)
 result={'status':'post-hoc HUST development; validation group-LOO only','runs':runs,'raw':regression_metrics(te['y'],np.mean(raw,0),te['groups']),'transported':regression_metrics(te['y'],np.mean(final,0),te['groups'])};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',raw=np.asarray(raw),transported=np.asarray(final),y=te['y'],groups=te['groups']);print(result['raw']['pooled']['r2'],result['transported']['pooled']['r2'])
if __name__=='__main__':main()
