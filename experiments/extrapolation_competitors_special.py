#!/usr/bin/env python3
"""NASA LOO and N-CMAPSS special-protocol extrapolation competitors."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'.benchmark_deps'),str(ROOT.parent/'ca-css-ncmapss')]
import extrapolation_competitors_matr as bench
from pp_extrapolation import regression_metrics,select_affine_initialization
OUT=ROOT/'results/extrapolation_competitors_all_v1';bench.OUT=OUT

def nasa():
 from run_affine_tail_external_nasa_health_v2 import prepare_folds
 folds,_=prepare_folds();result={}
 for kind in ('vrex','groupdro','monotone','linear_rff'):
  fold_results=[];pred=[];truth=[];groups=[]
  for i,f in enumerate(folds):
   parts=(f['train'],f['validation'],f['test']);aff=select_affine_initialization(*parts[:2]);prefix=f"nasa_fold{i}_"
   row=bench.linear_rff(parts,aff,prefix) if kind=='linear_rff' else bench.neural_model(kind,parts,aff,prefix)
   z=np.load(OUT/f'{prefix}{kind}_predictions.npz',allow_pickle=True);pred.append(z['prediction']);truth.append(z['y']);groups.append(z['groups']);fold_results.append({'test_cell':f['test_cell'],'result':row})
  matrix=np.concatenate(pred,axis=1);y=np.concatenate(truth);g=np.concatenate(groups);rs=[regression_metrics(y,p,g)['pooled']['r2'] for p in matrix]
  result[kind]={'folds':fold_results,'ensemble':regression_metrics(y,matrix.mean(0),g),'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs,ddof=1))}
 return result

def ncmapss():
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_benchmark import rows
 split=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);names=list(FEATURE_COLS);parts=(rows(split.train,names),rows(split.val,names),rows(split.test,names));aff=select_affine_initialization(*parts[:2]);result={}
 for kind in ('vrex','groupdro','monotone'):
  result[kind]=bench.neural_model(kind,parts,aff,'ncmapss_')
 result['linear_rff']=bench.linear_rff(parts,aff,'ncmapss_');return result

def main():
 torch.set_num_threads(2);path=OUT/'results.json';payload=json.load(open(path));payload['datasets']['nasa']=nasa();(OUT/'results.special.partial.json').write_text(json.dumps(payload,indent=2)+'\n');payload['datasets']['ncmapss']=ncmapss();path.write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
