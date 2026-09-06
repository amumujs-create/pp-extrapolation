#!/usr/bin/env python3
"""Post-hoc development audit of latent-transition PP."""
import json
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from regime_spline_deep_future import splits
SEEDS=(42,43,44,45,46);SEPARATION=(0.,.001,.01);GATE_WEIGHTS=(0.,.001,.01)
def run(tr,va,te):
 affine=select_affine_initialization(tr,va);rows=[];pred=[]
 for seed in SEEDS:
  candidates=[]
  for sw in SEPARATION:
   for gw in GATE_WEIGHTS:
    fit=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=affine,separation_weight=sw,gate_weight=gw)
    candidates.append((fit.selection['validation_mse'],sw,gw,fit))
  _,sw,gw,fit=min(candidates,key=lambda x:(x[0],x[1],x[2]));p,g=predict_latent_regime(fit,te['x'],return_gate=True);pred.append(p);m=regression_metrics(te['y'],p,te['groups'])
  rows.append({'seed':seed,'selected_separation':sw,'selected_gate_weight':gw,'selection':fit.selection,'test_gate':{'mean':float(g.mean()),'sd':float(g.std()),'min':float(g.min()),'max':float(g.max())},'metrics':m});print(seed,sw,gw,m['pooled']['r2'],g.mean(),flush=True)
 pred=np.asarray(pred);r=np.asarray([x['metrics']['pooled']['r2'] for x in rows]);return {'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std()),'ensemble':regression_metrics(te['y'],pred.mean(0),te['groups']),'runs':rows}
def main():
 torch.set_num_threads(2);data={}
 for name,split in splits().items():data[name]=run(*split)
 payload={'experiment':'latent_regime_deep_future_v1','status':'post-hoc doctoral extension development','architecture':'frozen affine + monotone latent transition gate + two learned neural tail experts','selection':'validation MSE only','gate_weights':GATE_WEIGHTS,'datasets':data}
 out=Path('results/latent_regime_deep_future_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
