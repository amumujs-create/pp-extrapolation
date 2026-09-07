#!/usr/bin/env python3
"""Linear-mean Gaussian process baseline on all positive PP splits."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,DotProduct,ConstantKernel
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import regression_metrics
from extrapolation_competitors_all import datasets
OUT=ROOT/'results/linear_mean_gp_all_positive_v1';SEEDS=range(42,47);GRID=[(l,n) for l in (.3,1.,3.) for n in (.01,.1,1.)]
def fit(tr,va,te,cfg,seed):
 rng=np.random.default_rng(seed);idx=np.arange(len(tr['y'])) if len(tr['y'])<=750 else np.sort(rng.choice(len(tr['y']),750,replace=False));x=tr['x'][idx].astype(float);y=tr['y'][idx].astype(float);mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);ym=y.mean();ys=max(y.std(),1e-8);z=(x-mu)/sd
 kernel=DotProduct(sigma_0=1.)+ConstantKernel(1.)*RBF(length_scale=cfg[0])+WhiteKernel(noise_level=cfg[1]);m=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=False).fit(z,(y-ym)/ys);cap=max(float(tr['y'].max()),1.)
 pred=lambda q:np.clip(m.predict((q-mu)/sd)*ys+ym,0,cap)
 vp=pred(va['x']);return float(np.mean((vp-va['y'])**2)),pred(te['x'])
def run(parts):
 tr,va,te=parts;search=[]
 for c in GRID:m,_=fit(tr,va,te,c,42);search.append({'config':c,'validation_mse':m})
 c=min(search,key=lambda z:z['validation_mse'])['config'];p=[]
 for s in SEEDS:_,q=fit(tr,va,te,c,s);p.append(q)
 return {'max_train_rows':750,'selected':c,'search':search,'ensemble':regression_metrics(te['y'],np.mean(p,0),te['groups'])}
def ncmapss():
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_benchmark import rows
 s=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);n=list(FEATURE_COLS);return rows(s.train,n),rows(s.val,n),rows(s.test,n)
def nasa():
 from run_affine_tail_external_nasa_health_v2 import prepare_folds
 fs,_=prepare_folds();pred=[[] for _ in SEEDS];ys=[];gs=[];audit=[]
 for f in fs:
  p=(f['train'],f['validation'],f['test']);search=[]
  for c in GRID:m,_=fit(*p,c,42);search.append({'config':c,'validation_mse':m})
  c=min(search,key=lambda z:z['validation_mse'])['config']
  for i,s in enumerate(SEEDS):_,q=fit(*p,c,s);pred[i].append(q)
  ys.append(p[2]['y']);gs.append(p[2]['groups']);audit.append({'test_cell':f['test_cell'],'selected':c})
 y=np.concatenate(ys);g=np.concatenate(gs);a=np.asarray([np.concatenate(x) for x in pred]);return {'folds':audit,'ensemble':regression_metrics(y,a.mean(0),g)}
def main():
 OUT.mkdir(parents=True,exist_ok=True);path=OUT/'results.json';r={'protocol':'linear mean + RBF GP; validation-only; max 750 train rows','datasets':{}};ds=datasets();ds={k:ds[k] for k in ('hust','virkler','sunwoda','rwth','matr','matr_batch2')};ds['ncmapss']=ncmapss()
 for n,p in ds.items():r['datasets'][n]=run(p);path.write_text(json.dumps(r,indent=2)+'\n');print(n,r['datasets'][n]['ensemble']['pooled']['r2'],flush=True)
 r['datasets']['nasa']=nasa();path.write_text(json.dumps(r,indent=2)+'\n');print('nasa',r['datasets']['nasa']['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
