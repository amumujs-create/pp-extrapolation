#!/usr/bin/env python3
"""Validation-only output calibration of archived latent PP on MATR2019."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from matr_2019_latent_confirmatory import load_cells,make_rows
from pp_extrapolation import (OutputCalibrator,approve_dual_evidence,
 group_loo_affine_evidence,fit_latent_regime_pp,predict_latent_regime,
 select_affine_initialization,regression_metrics)
OUT=ROOT/'results/matr_pp_validation_calibration_v1'

def group_bias(y,p,g):
 return float(np.mean([np.mean(y[g==u]-p[g==u]) for u in np.unique(g)]))

def fit_calibrator(kind,y,p,g):
 if kind=='identity':return (1.,0.)
 if kind=='bias':return (1.,float(np.mean(y-p)))
 if kind=='group_bias':return (1.,group_bias(y,p,g))
 if kind=='affine':
  # Restrict the slope to avoid unstable correction from eight validation cells.
  b,a=np.polyfit(p,y,1);return (float(np.clip(b,.5,1.5)),float(a))
 raise ValueError(kind)

def apply(params,p,cap):return np.clip(params[0]*p+params[1],0,cap)

def loo_score(kind,y,p,g,cap):
 errors=[]
 for held in np.unique(g):
  tr=g!=held;te=~tr;params=fit_calibrator(kind,y[tr],p[tr],g[tr]);errors.extend((apply(params,p[te],cap)-y[te])**2)
 return float(np.mean(errors))

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 source=json.load(open(ROOT/'results/matr_2019_latent_confirmatory/results.json'));audit=source['pretest'];cells={c['index']:c for c in load_cells()[0]}
 tr,va,te=[make_rows([cells[i] for i in audit['split_indices'][name]],audit['actual_boundary'],train=name=='train') for name in ('train','validation','test')]
 aff=select_affine_initialization(tr,va);vp=[];tp=[]
 for run in source['runs']['latent']:
  s=run['selection'];fit=fit_latent_regime_pp(tr,va,seed=s['seed'],affine_selection=aff,max_epochs=300,patience=70,
      separation_weight=run['separation'],gate_weight=run['gate_weight'])
  vp.append(predict_latent_regime(fit,va['x']));tp.append(predict_latent_regime(fit,te['x']));print('FIT',s['seed'],flush=True)
 vp=np.asarray(vp);tp=np.asarray(tp);cap=float(aff['target_scale']);kinds=('identity','bias','group_bias','affine');runs=[];pred=[]
 for i,seed in enumerate(range(42,47)):
  scores={k:loo_score(k,va['y'],vp[i],va['groups'],cap) for k in kinds};chosen=min(kinds,key=lambda k:scores[k]);params=fit_calibrator(chosen,va['y'],vp[i],va['groups']);p=apply(params,tp[i],cap);pred.append(p)
  runs.append({'seed':seed,'loo_validation_mse':scores,'chosen':chosen,'parameters':params,'validation_bias':float(np.mean(vp[i]-va['y'])),
               'metrics':regression_metrics(te['y'],p,te['groups'])});print(seed,chosen,params,runs[-1]['metrics']['pooled']['r2'],flush=True)
 evidence=group_loo_affine_evidence(va['y'],vp.mean(0),va['groups'],upper=cap)
 approved=approve_dual_evidence([OutputCalibrator(run['chosen']) for run in runs],evidence)
 ens=regression_metrics(te['y'],np.mean(pred,0),te['groups']);base=regression_metrics(te['y'],tp.mean(0),te['groups'])
 final=ens if approved else base
 result={'status':'post-hoc development; calibrator selected by leave-one-validation-cell-out only','candidates':kinds,'runs':runs,
         'group_evidence':evidence,'dual_evidence_approved':approved,
         'base_ensemble':base,'calibrated_ensemble':ens,'final_ensemble':final}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=np.asarray(pred),base=tp,y=te['y'],groups=te['groups'],validation_prediction=vp,validation_y=va['y'],validation_groups=va['groups'])
 print('DONE',base['pooled']['r2'],ens['pooled']['r2'],approved)
if __name__=='__main__':main()
