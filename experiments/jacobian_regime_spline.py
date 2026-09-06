#!/usr/bin/env python3
"""Validation-selected total-output Jacobian penalty for deep-future PP."""
import json
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_regime_spline_pp,jacobian_diagnostics,predict_regime_spline,regression_metrics,select_affine_initialization
from regime_spline_deep_future import splits
SEEDS=(42,43,44,45,46);WEIGHTS=(0.,.01,.1,1.)
def run(tr,va,te):
 affine=select_affine_initialization(tr,va);runs=[];pred=[]
 for seed in SEEDS:
  candidates=[]
  for weight in WEIGHTS:
   fit=fit_regime_spline_pp(tr,va,seed=seed,affine_selection=affine,max_epochs=300,jacobian_weight=weight,jacobian_ray_multiplier=1.)
   candidates.append((fit.selection['validation_mse'],weight,fit))
  _,weight,fit=min(candidates,key=lambda x:(x[0],x[1]));p=predict_regime_spline(fit,te['x']);m=regression_metrics(te['y'],p,te['groups']);pred.append(p)
  runs.append({'seed':seed,'selected_weight':weight,'candidate_validation_mse':{str(w):float(v) for v,w,_ in candidates},'selection':fit.selection,'test_jacobian':jacobian_diagnostics(fit,te['x']),'metrics':m});print(seed,weight,m['pooled']['r2'],flush=True)
 pred=np.asarray(pred);r=np.array([x['metrics']['pooled']['r2'] for x in runs]);return {'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std()),'ensemble':regression_metrics(te['y'],pred.mean(0),te['groups']),'runs':runs}
def main():
 torch.set_num_threads(2);base=json.load(open('results/regime_spline_deep_future_v1/results.json'))['datasets'];data={}
 for n,s in splits().items():data[n]={'free_spline':base[n]['regime_spline_pp'],'jacobian_spline':run(*s)}
 payload={'experiment':'jacobian_regime_spline_v1','status':'post-hoc deep-future development','weights':WEIGHTS,'penalty':'relu(d yhat / d oriented_degradation)^2 on total output along label-free counterfactual future rays','ray_multiplier':1.,'datasets':data}
 out=Path('results/jacobian_regime_spline_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
