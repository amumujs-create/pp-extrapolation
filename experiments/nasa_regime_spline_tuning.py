#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_regime_spline_pp,predict_regime_spline,regression_metrics,select_affine_initialization
from run_affine_tail_external_nasa_health_v2 import prepare_folds
OUT=ROOT/'results/nasa_regime_spline_tuning_v1';SEEDS=range(42,47);GRID=[(m,j,r) for m in (False,True) for j in (0.,.01,.1,1.) for r in (0.,1.,2.) if (j>0)==(r>0)]
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);folds,_=prepare_folds();pred=[[] for _ in SEEDS];ys=[];gs=[];audit=[]
 for f in folds:
  tr,va,te=f['train'],f['validation'],f['test'];a=select_affine_initialization(tr,va);search=[]
  for c in GRID:
   q=fit_regime_spline_pp(tr,va,seed=42,affine_selection=a,max_epochs=400,patience=80,monotone=c[0],jacobian_weight=c[1],jacobian_ray_multiplier=c[2]);search.append({'config':c,'mse':q.selection['validation_mse']})
  c=min(search,key=lambda x:x['mse'])['config']
  for i,s in enumerate(SEEDS):q=fit_regime_spline_pp(tr,va,seed=s,affine_selection=a,max_epochs=400,patience=80,monotone=c[0],jacobian_weight=c[1],jacobian_ray_multiplier=c[2]);pred[i].append(predict_regime_spline(q,te['x']))
  ys.append(te['y']);gs.append(te['groups']);audit.append({'test_cell':f['test_cell'],'selected':c,'search':search});print(f['test_cell'],c,flush=True)
 y=np.concatenate(ys);g=np.concatenate(gs);p=np.asarray([np.concatenate(x) for x in pred]);r={'folds':audit,'ensemble':regression_metrics(y,p.mean(0),g)};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=p,y=y,groups=g);print(r['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
