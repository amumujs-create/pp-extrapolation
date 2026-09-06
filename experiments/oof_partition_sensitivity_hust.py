#!/usr/bin/env python3
"""Sensitivity of the OOF-UQ PP result to nested partitions and head seeds."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import (combine_uncertainty_gated,fit_pp,fit_uncertainty_head,
 nested_oof_disagreement,predict,predict_uncertainty,regression_metrics,select_affine_initialization,support_distance)
from pp_extrapolation.support_gate import predict_components
from oof_uncertainty_pp import choose,components

ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from run_affine_tail_external_three import prepare_hust

def main():
 torch.set_num_threads(2);tr,va,te,_=prepare_hust();aff=select_affine_initialization(tr,va)
 fits=[fit_pp(tr,va,seed=s,affine_selection=aff,max_epochs=300) for s in (42,43,44,45,46)]
 pp=np.asarray([predict(f,te['x']) for f in fits]);vd,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);td,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
 a,r,cap=components(fits,te['x']);rows=[]
 for split_seed in (911,912,913,914,915):
  oof=nested_oof_disagreement(tr,max_epochs=200,split_seed=split_seed);head=fit_uncertainty_head(tr,oof,seed=split_seed+1000)
  vu=predict_uncertainty(head,va['x']);tu=predict_uncertainty(head,te['x']);sel,_=choose(fits,va['x'],va['y'],vd,vu)
  pred=combine_uncertainty_gated(a,r,td,tu,beta=sel['beta'],gamma=sel['gamma'],output_cap=cap)
  scores=[regression_metrics(te['y'],x,te['groups'])['pooled']['r2'] for x in pred]
  rows.append({'split_seed':split_seed,'head_seed':split_seed+1000,'selection':sel,'pooled_r2_mean':float(np.mean(scores)),'pooled_r2_sd':float(np.std(scores)),'per_model_seed':scores})
  print(split_seed,sel,float(np.mean(scores)),flush=True)
 base=[regression_metrics(te['y'],x,te['groups'])['pooled']['r2'] for x in pp]
 payload={'dataset':'HUST','status':'development_partition_sensitivity','baseline_pp':{'mean':float(np.mean(base)),'sd':float(np.std(base))},'runs':rows,
 'summary':{'mean_over_partitions':float(np.mean([x['pooled_r2_mean'] for x in rows])),'sd_over_partitions':float(np.std([x['pooled_r2_mean'] for x in rows])),
 'improved_partition_count':int(sum(x['pooled_r2_mean']>np.mean(base)+1e-6 for x in rows))}}
 out=Path('results/oof_partition_sensitivity_hust_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
