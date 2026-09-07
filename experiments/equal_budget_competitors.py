#!/usr/bin/env python3
"""Paper-grade staged tuning of neural competitors on positive-R2 settings."""
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'.benchmark_deps'),str(ROOT.parent/'ca-css-ncmapss')]
import extrapolation_competitors_matr as b
from extrapolation_competitors_all import datasets
from pp_extrapolation import regression_metrics,select_affine_initialization
OUT=ROOT/'results/equal_budget_competitors_v1';KINDS=('vrex','groupdro','monotone');SEEDS=range(42,47)
ARCH=[(w,d,lr,wd,.01) for w,d in ((16,1),(32,1),(32,2),(64,2)) for lr in (2e-4,5e-4,1e-3) for wd in (.1,2.)]
PEN=(.001,.01,.1,1.,10.)
def ncmapss():
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_benchmark import rows
 s=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);n=list(FEATURE_COLS);return rows(s.train,n),rows(s.val,n),rows(s.test,n)
def tune(name,kind,parts):
 aff=select_affine_initialization(parts[0],parts[1]);search=[]
 for cfg in ARCH:
  info,_=b.fit_neural(kind,cfg,42,parts,aff);search.append({'config':cfg,**info})
 base=min(search,key=lambda z:z['validation_mse'])['config'];stage=[(*base[:4],p) for p in PEN]
 for cfg in stage:
  info,_=b.fit_neural(kind,cfg,42,parts,aff);search.append({'config':cfg,**info})
 chosen=min(search,key=lambda z:z['validation_mse'])['config'];runs=[];pred=[]
 for s in SEEDS:
  info,p=b.fit_neural(kind,chosen,s,parts,aff,True);pred.append(p);runs.append({'seed':s,**info,'metrics':regression_metrics(parts[2]['y'],p,parts[2]['groups'])})
 return {'search_budget':len(search),'selected_config':chosen,'search':search,'runs':runs,'ensemble':regression_metrics(parts[2]['y'],np.mean(pred,0),parts[2]['groups'])}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);path=OUT/'results.json';r=json.load(open(path)) if path.exists() else {'protocol':'24 architecture/optimizer candidates + 5 penalty candidates; validation only','datasets':{}}
 ds=datasets();ds={k:ds[k] for k in ('hust','virkler','sunwoda','rwth','matr','matr_batch2')};ds['ncmapss']=ncmapss()
 for name,parts in ds.items():
  r['datasets'].setdefault(name,{})
  for kind in KINDS:
   if kind in r['datasets'][name]:continue
   started=time.time();r['datasets'][name][kind]=tune(name,kind,parts);r['datasets'][name][kind]['seconds']=time.time()-started;path.write_text(json.dumps(r,indent=2)+'\n');print('DONE',name,kind,r['datasets'][name][kind]['ensemble']['pooled']['r2'],flush=True)
if __name__=='__main__':main()
