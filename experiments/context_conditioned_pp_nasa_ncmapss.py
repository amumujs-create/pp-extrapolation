#!/usr/bin/env python3
"""Validation-only context-conditioned PP output head for NASA and N-CMAPSS.

The same module is used in both domains: identity PP is always a candidate; otherwise
ridge estimates a residual correction from PP prediction and standardized causal
context. Group LOO is preferred; a contiguous blocked CV is used with one validation unit.
"""
from pathlib import Path
import json,sys
import numpy as np,torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_regime_spline_pp,predict_regime_spline,fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
SEEDS=range(42,47);ALPHAS=(.1,1.,10.,100.,1000.);OUT=ROOT/'results/context_conditioned_pp_nasa_ncmapss_v1'

def folds(groups,n):
 u=np.unique(groups)
 if len(u)>=3:return [np.flatnonzero(groups==x) for x in u]
 order=np.arange(n);return [x for x in np.array_split(order,4) if len(x)]
def cols_for(x):
 # Fixed generic summaries: prediction alone, first coordinate, short context, all context.
 return {'identity':(), 'coordinate':(0,), 'short':tuple(range(min(4,x.shape[1]))), 'all':tuple(range(x.shape[1]))}
def design(p,x,cols,mu=None,sd=None):
 if not cols:return p[:,None],None,None
 z=x[:,cols]
 if mu is None:mu=z.mean(0);sd=np.maximum(z.std(0),1e-6)
 return np.c_[p,(z-mu)/sd],mu,sd
def cv_mse(p,x,y,g,kind,alpha,cap):
 if kind=='identity':return float(np.mean((p-y)**2))
 err=[];cols=cols_for(x)[kind]
 for hold in folds(g,len(y)):
  keep=np.ones(len(y),bool);keep[hold]=False;A,mu,sd=design(p[keep],x[keep],cols);B,_,_=design(p[hold],x[hold],cols,mu,sd)
  q=np.clip(Ridge(alpha=alpha).fit(A,y[keep]).predict(B),0,cap);err.extend((q-y[hold])**2)
 return float(np.mean(err))
def select_apply(vp,vx,vy,vg,tp,tx,cap):
 cand=[{'kind':'identity','alpha':None,'cv_mse':cv_mse(vp,vx,vy,vg,'identity',1,cap)}]
 for kind in ('coordinate','short','all'):
  for a in ALPHAS:cand.append({'kind':kind,'alpha':a,'cv_mse':cv_mse(vp,vx,vy,vg,kind,a,cap)})
 best=min(cand,key=lambda z:z['cv_mse'])
 if best['kind']=='identity':return tp,best,cand
 cols=cols_for(vx)[best['kind']];A,mu,sd=design(vp,vx,cols);B,_,_=design(tp,tx,cols,mu,sd)
 return np.clip(Ridge(alpha=best['alpha']).fit(A,vy).predict(B),0,cap),best,cand

def nasa():
 from run_affine_tail_external_nasa_health_v2 import prepare_folds
 archived=json.load(open(ROOT/'results/nasa_regime_spline_tuning_v1/results.json'));fs,_=prepare_folds();raw=[[] for _ in SEEDS];final=[[] for _ in SEEDS];ys=[];gs=[];audit=[]
 for fi,f in enumerate(fs):
  tr,va,te=f['train'],f['validation'],f['test'];cfg=archived['folds'][fi]['selected'];aff=select_affine_initialization(tr,va);runs=[]
  for j,s in enumerate(SEEDS):
   q=fit_regime_spline_pp(tr,va,seed=s,affine_selection=aff,max_epochs=400,patience=80,monotone=cfg[0],jacobian_weight=cfg[1],jacobian_ray_multiplier=cfg[2]);vp=predict_regime_spline(q,va['x']);tp=predict_regime_spline(q,te['x']);z,b,c=select_apply(vp,va['x'],va['y'],va['groups'],tp,te['x'],q.target_scale);raw[j].append(tp);final[j].append(z);runs.append({'seed':s,'selected_head':b,'candidates':c})
  ys.append(te['y']);gs.append(te['groups']);audit.append({'test_cell':f['test_cell'],'base_config':cfg,'runs':runs});print('NASA',f['test_cell'],flush=True)
 y=np.concatenate(ys);g=np.concatenate(gs);a=np.asarray([np.concatenate(x) for x in raw]);b=np.asarray([np.concatenate(x) for x in final]);return {'folds':audit,'raw':regression_metrics(y,a.mean(0),g),'context_pp':regression_metrics(y,b.mean(0),g),'seed_metrics':[regression_metrics(y,x,g) for x in b]}

def ncmapss():
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_css import eval_all_bands_hard
 from ncmapss_pp_multiscale import make_features,as_rows
 split=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);names=list(FEATURE_COLS);arch=json.load(open(ROOT/'results/ncmapss_pp_multiscale_v1/results.json'));raw=[];final=[];runs=[]
 for j,s in enumerate(SEEDS):
  sel=arch['runs'][j]['selected'];preset=sel['preset'];tr=as_rows(split.train,names,preset);va=as_rows(split.val,names,preset);aff=select_affine_initialization(tr,va);q=fit_latent_regime_pp(tr,va,seed=s,affine_selection=aff,max_epochs=300,patience=70,separation_weight=sel['separation_weight'],gate_weight=0.)
  allx=make_features(split.all_windows,names,preset);vp=predict_latent_regime(q,va['x']);tp=predict_latent_regime(q,allx);z,b,c=select_apply(vp,va['x'],va['y'],va['groups'],tp,allx,q.target_scale);raw.append(tp);final.append(z);runs.append({'seed':s,'base_config':sel,'selected_head':b,'candidates':c});print('NCMAPSS',s,b,flush=True)
 a=np.mean(raw,0);b=np.mean(final,0);return {'runs':runs,'raw':eval_all_bands_hard(split,a)['hard_extrap'],'context_pp':eval_all_bands_hard(split,b)['hard_extrap']}
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True);r={'status':'post-hoc targeted development; identical validation-only context head in both domains','module':'identity-or-ridge residual correction using causal context; group LOO or contiguous blocked validation CV','nasa':nasa(),'ncmapss':ncmapss()};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print('FINAL',r['nasa']['raw']['pooled']['r2'],r['nasa']['context_pp']['pooled']['r2'],r['ncmapss']['raw']['overall']['r2'],r['ncmapss']['context_pp']['overall']['r2'])
if __name__=='__main__':main()
