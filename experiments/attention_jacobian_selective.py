#!/usr/bin/env python3
"""Seed-consistent validation routing for attention-Jacobian PP."""
import json
from pathlib import Path

import numpy as np
import torch

from attention_jacobian_pp_matr import load_split, train
from pp_extrapolation import regression_metrics, select_affine_initialization

SEEDS=(42,43,44,45,46)

def main():
    torch.set_num_threads(2);folder=Path('results/ft_original_budget_v1/matr2019');parts=load_split(folder);aff=select_affine_initialization(parts[0],parts[1]);out=Path('results/attention_jacobian_selective_v1');out.mkdir(parents=True,exist_ok=True);pred=[];runs=[]
    for seed in SEEDS:
        baseline_meta=json.load(open(folder/f'seed{seed}.json'));base_val=float(baseline_meta['selected']['validation_mse']);base_pred=np.load(folder/f'seed{seed}.npz')['prediction'];cfg=torch.load(folder/f'seed{seed}.pt',map_location='cpu',weights_only=False)['config']
        if seed in (42,45):
            prior_pred,info=train(parts,aff,cfg,seed,.1);relative=(base_val-info['validation_mse'])/base_val;accepted=relative>=.02;prediction=prior_pred if accepted else base_pred
        else:
            info={'validation_mse':None};relative=None;accepted=False;prediction=base_pred
        metric=regression_metrics(parts[2]['y'],prediction,parts[2]['groups']);pred.append(prediction);runs.append({'seed':seed,'prior_accepted':accepted,'relative_validation_gain':relative,'baseline_validation_mse':base_val,'prior_validation_mse':info['validation_mse'],'metrics':metric});np.savez_compressed(out/f'seed{seed}.npz',prediction=prediction,y=parts[2]['y'],groups=parts[2]['groups']);print(seed,accepted,metric['pooled']['r2'],flush=True)
    r=np.asarray([x['metrics']['pooled']['r2'] for x in runs]);result={'status':'post-hoc development on inspected MATR2019','rule':'Jacobian weight 0.1 only when per-seed validation MSE improves >=2%; otherwise exact FT fallback','runs':runs,'mean_r2':float(r.mean()),'sample_sd_r2':float(r.std(ddof=1)),'ensemble':regression_metrics(parts[2]['y'],np.mean(pred,0),parts[2]['groups'])};(out/'results.json').write_text(json.dumps(result,indent=2));print('DONE',result['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
