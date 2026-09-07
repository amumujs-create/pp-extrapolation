#!/usr/bin/env python3
"""Group-LOO state/rate transport for validation-selected robust PP on MATR batch2."""
import json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from group_robust_pp import batch2
OUT=ROOT/'results/matr_batch2_regime_transport_pp_v1';SEEDS=range(42,47);ALPHAS=(.1,1.,10.,100.,1000.)
def des(kind,p,x,mu=None,sd=None):
 cols={'affine':(), 'state':(0,1),'rate':(2,3),'all':tuple(range(x.shape[1]))}[kind];z=x[:,cols] if cols else np.empty((len(x),0))
 if mu is None:mu=z.mean(0);sd=np.maximum(z.std(0),1e-8)
 return np.c_[p,(z-mu)/sd],mu,sd
def loo(k,a,y,p,x,g,cap):
 e=[]
 for u in np.unique(g):
  q=g!=u;A,mu,sd=des(k,p[q],x[q]);B,_,_=des(k,p[~q],x[~q],mu,sd);m=Ridge(alpha=a).fit(A,y[q]);e.extend((np.clip(m.predict(B),0,cap)-y[~q])**2)
 return np.mean(e)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);tr,va,te=batch2();aff=select_affine_initialization(tr,va);raw=[];out=[];audit=[]
 for s in SEEDS:
  f=fit_pp(tr,va,seed=s,affine_selection=aff,max_epochs=300,patience=70,width=64,group_dro_eta=1.,residual_decay=.1);vp=predict(f,va['x']);tp=predict(f,te['x']);cs=[{'kind':k,'alpha':a,'mse':float(loo(k,a,va['y'],vp,va['x'],va['groups'],f.target_scale))} for k in ('affine','state','rate','all') for a in ALPHAS];z=min(cs,key=lambda x:x['mse']);A,mu,sd=des(z['kind'],vp,va['x']);B,_,_=des(z['kind'],tp,te['x'],mu,sd);q=np.clip(Ridge(alpha=z['alpha']).fit(A,va['y']).predict(B),0,f.target_scale);raw.append(tp);out.append(q);audit.append({'seed':s,'chosen':z});print(s,z,flush=True)
 r={'selection':'validation group-LOO only','runs':audit,'raw':regression_metrics(te['y'],np.mean(raw,0),te['groups']),'transported':regression_metrics(te['y'],np.mean(out,0),te['groups'])};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['transported']['pooled']['r2'])
if __name__=='__main__':main()
