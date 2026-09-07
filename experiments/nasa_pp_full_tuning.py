#!/usr/bin/env python3
"""Staged validation-loss PP tuning for NASA battery leave-one-cell-out folds."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from run_affine_tail_external_nasa_health_v2 import prepare_folds
OUT=ROOT/'results/nasa_pp_full_tuning_v1';SEEDS=range(42,47)
BASE=[{'width':w,'lr':lr,'wd':wd,'eta':0.,'decay':0.,'anchor':None} for w in (16,32,64) for lr in (2e-4,5e-4,1e-3) for wd in (.1,2.)]
def train(parts,c,s):
 a=select_affine_initialization(parts[0],parts[1]);return fit_pp(parts[0],parts[1],seed=s,affine_selection=a,max_epochs=400,patience=80,width=c['width'],learning_rate=c['lr'],weight_decay=c['wd'],group_dro_eta=c['eta'],residual_decay=c['decay'],affine_anchor_weight=c['anchor'])
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);folds,_=prepare_folds();allp=[[] for _ in SEEDS];ys=[];gs=[];audit=[]
 for fi,f in enumerate(folds):
  parts=(f['train'],f['validation'],f['test']);search=[]
  for c in BASE:
   m=train(parts,c,42);search.append({'config':c,'mse':m.selection['best_validation_mse']})
  best=min(search,key=lambda z:z['mse'])['config'];stage=[{**best,'eta':e,'decay':d} for e in (0.,.001,.01,.1) for d in (0.,.1)]+[{**best,'anchor':a} for a in (.1,1.,10.)]
  for c in stage:
   m=train(parts,c,42);search.append({'config':c,'mse':m.selection['best_validation_mse']})
  chosen=min(search,key=lambda z:z['mse'])['config'];runs=[]
  for si,s in enumerate(SEEDS):
   m=train(parts,chosen,s);p=predict(m,parts[2]['x']);allp[si].append(p);runs.append({'seed':s,'validation_mse':m.selection['best_validation_mse'],'epoch':m.selection['selected_epoch']})
  ys.append(parts[2]['y']);gs.append(parts[2]['groups']);audit.append({'test_cell':f['test_cell'],'selected':chosen,'search':search,'runs':runs});print(f['test_cell'],chosen,flush=True)
 y=np.concatenate(ys);g=np.concatenate(gs);matrix=np.asarray([np.concatenate(x) for x in allp]);result={'selection':'staged validation MSE only','folds':audit,'ensemble':regression_metrics(y,matrix.mean(0),g),'per_seed':[regression_metrics(y,p,g) for p in matrix]};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=matrix,y=y,groups=g);print(result['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
