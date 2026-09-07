#!/usr/bin/env python3
"""Equal-search-budget PP tuning plus group-LOO transport on MATR batch2."""
import json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from group_robust_pp import batch2
from matr_batch2_regime_transport_pp import des,loo,ALPHAS
OUT=ROOT/'results/matr_batch2_pp_equal_tuning_v1';SEEDS=range(42,47)
BASE=[{'width':w,'lr':lr,'wd':wd,'eta':0.,'decay':0.} for w in (16,32,64) for lr in (2e-4,5e-4,1e-3) for wd in (.1,2.)]
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);tr,va,te=batch2();aff=select_affine_initialization(tr,va)
 def fit(c,s):return fit_pp(tr,va,seed=s,affine_selection=aff,max_epochs=400,patience=80,width=c['width'],learning_rate=c['lr'],weight_decay=c['wd'],group_dro_eta=c['eta'],residual_decay=c['decay'])
 search=[]
 for c in BASE:
  f=fit(c,42);search.append({'config':c,'mse':f.selection['best_validation_mse']})
 base=min(search,key=lambda x:x['mse'])['config']
 for e in (0.,.001,.01,.1,1.):
  for d in (0.,.05,.1):
   c={**base,'eta':e,'decay':d};f=fit(c,42);search.append({'config':c,'mse':f.selection['best_validation_mse']})
 chosen=min(search,key=lambda x:x['mse'])['config'];raw=[];out=[];runs=[]
 for s in SEEDS:
  f=fit(chosen,s);vp=predict(f,va['x']);tp=predict(f,te['x']);cs=[{'kind':k,'alpha':a,'mse':float(loo(k,a,va['y'],vp,va['x'],va['groups'],f.target_scale))} for k in ('affine','state','rate','all') for a in ALPHAS];z=min(cs,key=lambda x:x['mse']);A,mu,sd=des(z['kind'],vp,va['x']);B,_,_=des(z['kind'],tp,te['x'],mu,sd);q=np.clip(Ridge(alpha=z['alpha']).fit(A,va['y']).predict(B),0,f.target_scale);raw.append(tp);out.append(q);runs.append({'seed':s,'transport':z})
 r={'search_budget':len(search),'selected':chosen,'search':search,'runs':runs,'raw':regression_metrics(te['y'],np.mean(raw,0),te['groups']),'transported':regression_metrics(te['y'],np.mean(out,0),te['groups'])};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(chosen,r['raw']['pooled']['r2'],r['transported']['pooled']['r2'])
if __name__=='__main__':main()
