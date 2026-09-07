#!/usr/bin/env python3
"""Official Engression benchmark on all positive-R2 PP extrapolation splits."""
import json,os,sys,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/pp-mpl')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.benchmark_deps'),str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
import numpy as np,torch
from engression import engression
from pp_extrapolation import regression_metrics
from extrapolation_competitors_all import datasets
OUT=ROOT/'results/engression_all_positive_v2';SEEDS=range(42,47);GRID=[(h,lr,b,layers,epochs) for h in (32,64) for lr in (1e-3,5e-3) for b in (.5,1.) for layers,epochs in ((2,250),(3,500))]
def prep(part,idx=None):
 x=part['x'] if idx is None else part['x'][idx];y=part['y'] if idx is None else part['y'][idx];return torch.tensor(x,dtype=torch.float32),torch.tensor(y[:,None],dtype=torch.float32)
def fit_predict(tr,va,te,cfg,seed):
 rng=np.random.default_rng(seed);idx=np.arange(len(tr['y'])) if len(tr['y'])<=5000 else np.sort(rng.choice(len(tr['y']),5000,replace=False));x,y=prep(tr,idx);vx,vy=prep(va);tx,_=prep(te);torch.manual_seed(seed)
 h,lr,beta,layers,epochs=cfg
 m=engression(x,y,num_layer=layers,hidden_dim=h,noise_dim=32,beta=beta,lr=lr,num_epoches=epochs,batch_size=min(512,len(x)),device='cpu',standardize=True,verbose=False)
 with torch.no_grad():vp=m.predict(vx,target='mean',sample_size=50).squeeze().cpu().numpy();tp=m.predict(tx,target='mean',sample_size=100).squeeze().cpu().numpy()
 cap=max(float(tr['y'].max()),1.);return float(np.mean((np.clip(vp,0,cap)-va['y'])**2)),np.clip(tp,0,cap)
def run(parts):
 tr,va,te=parts;search=[]
 for c in GRID:
  mse,_=fit_predict(tr,va,te,c,42);search.append({'config':c,'validation_mse':mse});print(c,mse,flush=True)
 c=min(search,key=lambda x:x['validation_mse'])['config'];pred=[];runs=[]
 for s in SEEDS:
  mse,p=fit_predict(tr,va,te,c,s);pred.append(p);runs.append({'seed':s,'validation_mse':mse,'metrics':regression_metrics(te['y'],p,te['groups'])})
 return {'official_package_version':'0.1.9','max_train_rows':5000,'selected':c,'search':search,'runs':runs,'ensemble':regression_metrics(te['y'],np.mean(pred,0),te['groups'])}
def ncmapss():
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_benchmark import rows
 s=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);n=list(FEATURE_COLS);return rows(s.train,n),rows(s.val,n),rows(s.test,n)
def nasa():
 from run_affine_tail_external_nasa_health_v2 import prepare_folds
 folds,_=prepare_folds();truth=[];groups=[];matrix=[[] for _ in SEEDS];audit=[]
 for f in folds:
  parts=(f['train'],f['validation'],f['test']);search=[]
  for c in GRID:
   mse,_=fit_predict(*parts,c,42);search.append({'config':c,'validation_mse':mse})
  c=min(search,key=lambda x:x['validation_mse'])['config']
  for i,s in enumerate(SEEDS):_,p=fit_predict(*parts,c,s);matrix[i].append(p)
  truth.append(parts[2]['y']);groups.append(parts[2]['groups']);audit.append({'test_cell':f['test_cell'],'selected':c,'search':search})
 y=np.concatenate(truth);g=np.concatenate(groups);p=np.asarray([np.concatenate(x) for x in matrix]);return {'folds':audit,'ensemble':regression_metrics(y,p.mean(0),g)}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);path=OUT/'results.json';r=json.load(open(path)) if path.exists() else {'protocol':'official Engression; validation-only hyperparameters; 5 seeds','datasets':{}};ds=datasets();ds={k:ds[k] for k in ('hust','virkler','sunwoda','rwth','matr','matr_batch2')};ds['ncmapss']=ncmapss()
 for n,p in ds.items():
  if n in r['datasets']:continue
  st=time.time();r['datasets'][n]=run(p);r['datasets'][n]['seconds']=time.time()-st;path.write_text(json.dumps(r,indent=2)+'\n');print('DONE',n,r['datasets'][n]['ensemble']['pooled']['r2'],flush=True)
 if 'nasa' not in r['datasets']:r['datasets']['nasa']=nasa();path.write_text(json.dumps(r,indent=2)+'\n');print('DONE nasa',r['datasets']['nasa']['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
