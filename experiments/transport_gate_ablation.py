#!/usr/bin/env python3
"""MATR correction-family and seed/unit/dual evidence ablation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pp_extrapolation import (consensus_tail_probability,fit_output_calibrator,
                              regression_metrics)

ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'pp':ROOT/'results/matr_pp_validation_calibration_v1/predictions.npz',
 'ft':ROOT/'results/matr_ft_validation_calibration_v1/predictions.npz',
}

def loo_losses(kind,y,p,g,cap):
 rows=[]
 for held in np.unique(g):
  keep=g!=held;cal=fit_output_calibrator(kind,y[keep],p[keep],g[keep])
  rows.append(float(np.mean((cal.predict(p[~keep],upper=cap)-y[~keep])**2)))
 return np.asarray(rows)

def family_evidence(kind,y,pred,g,cap,draws=20000):
 identity=[];candidate=[]
 for seed,p in enumerate(pred):
  a=loo_losses('identity',y,p,g,cap);b=loo_losses(kind,y,p,g,cap)
  identity.append(a);candidate.append(b)
 identity=np.asarray(identity);candidate=np.asarray(candidate)
 seed_gain=identity.mean(1)-candidate.mean(1);seed_wins=int(np.sum(seed_gain>0))
 # Physical-unit evidence is computed from the ensemble prediction, with the
 # calibrator refit while each validation unit is held out.
 a=loo_losses('identity',y,pred.mean(0),g,cap);b=loo_losses(kind,y,pred.mean(0),g,cap)
 group_gain=a-b;rng=np.random.default_rng(314159)
 boot=group_gain[rng.integers(0,len(group_gain),(draws,len(group_gain)))].mean(1)
 ci=np.quantile(boot,[.025,.975]);seed_p=consensus_tail_probability(seed_wins,len(pred))
 return {'seed_wins':seed_wins,'seed_total':len(pred),'seed_p':seed_p,
         'seed_approved':bool(seed_p<=.05),'group_count':len(group_gain),
         'group_wins':int(np.sum(group_gain>0)),'mean_group_mse_gain':float(group_gain.mean()),
         'group_bootstrap_ci':[float(ci[0]),float(ci[1])],
         'unit_approved':bool(ci[0]>0)}

def main():
 split=np.load(ROOT/'results/ft_original_budget_v1/matr2019/split.npz')
 cap=float(np.max(split['train_y']))
 result={'status':'post-hoc method ablation on the fixed MATR split','models':{}}
 for model,path in MODELS.items():
  z=np.load(path,allow_pickle=True);vp=z['validation_prediction'];vy=z['validation_y'];vg=z['validation_groups']
  base=z['base'];y=z['y'];groups=z['groups']
  base_metric=regression_metrics(y,base.mean(0),groups);families={}
  for kind in ('bias','group_bias','affine'):
   evidence=family_evidence(kind,vy,vp,vg,cap)
   calibrated=[]
   for v,t in zip(vp,base):
    cal=fit_output_calibrator(kind,vy,v,vg);calibrated.append(cal.predict(t,upper=cap))
   calibrated=np.asarray(calibrated);cal_metric=regression_metrics(y,calibrated.mean(0),groups)
   seed_only=evidence['seed_approved'];unit_only=evidence['unit_approved'];dual=seed_only and unit_only
   families[kind]={'evidence':evidence,'unconditional':cal_metric,
    'routes':{
     'seed_only':cal_metric if seed_only else base_metric,
     'unit_only':cal_metric if unit_only else base_metric,
     'dual':cal_metric if dual else base_metric}}
  result['models'][model]={'base':base_metric,'families':families}
 out=ROOT/'results/transport_gate_ablation_v1';out.mkdir(parents=True,exist_ok=True)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 for model,v in result['models'].items():
  print(model,'base',v['base']['pooled']['r2'])
  for kind,row in v['families'].items():
   e=row['evidence'];print(kind,'seed',e['seed_wins'],e['seed_p'],'unit_ci',e['group_bootstrap_ci'],
    'raw',row['unconditional']['pooled']['r2'],'dual',row['routes']['dual']['pooled']['r2'])

if __name__=='__main__':main()
