#!/usr/bin/env python3
"""Apply the identical validation-only calibration protocol to stored FT models."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'.benchmark_deps'))
from extended_nn_benchmark import Model
from matr_pp_validation_calibration import fit_calibrator,loo_score,apply
from pp_extrapolation import (OutputCalibrator,approve_dual_evidence,
 group_loo_affine_evidence,regression_metrics,select_affine_initialization)
from pp_extrapolation.model import transform_features
OUT=ROOT/'results/matr_ft_validation_calibration_v1';BASE=ROOT/'results/ft_original_budget_v1/matr2019'
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);a=np.load(BASE/'split.npz')
 parts=[{k:a[f'{label}_{k}'] for k in ('x','y','groups')} for label in ('train','validation','test')];tr,va,te=parts
 aff=select_affine_initialization(tr,va);z=transform_features(tr['x'],aff['center'],aff['scale']);vz=transform_features(va['x'],aff['center'],aff['scale']);tz=transform_features(te['x'],aff['center'],aff['scale']);cap=float(aff['target_scale'])
 kinds=('identity','bias','group_bias','affine');vp=[];tp=[];runs=[];pred=[]
 for seed in range(42,47):
  ck=torch.load(BASE/f'seed{seed}.pt',map_location='cpu',weights_only=False);model=Model('ft_transformer',z.shape[1],ck['config'],aff,z,vz);model.load_state_dict(ck['state']);model.eval()
  with torch.no_grad():v=model(torch.tensor(vz)).numpy()*cap;t=model(torch.tensor(tz)).numpy()*cap
  v=np.clip(v,0,cap);t=np.clip(t,0,cap);vp.append(v);tp.append(t);scores={k:loo_score(k,va['y'],v,va['groups'],cap) for k in kinds};chosen=min(kinds,key=lambda k:scores[k]);params=fit_calibrator(chosen,va['y'],v,va['groups']);p=apply(params,t,cap);pred.append(p)
  runs.append({'seed':seed,'loo_validation_mse':scores,'chosen':chosen,'parameters':params,'metrics':regression_metrics(te['y'],p,te['groups'])});print(seed,chosen,params,runs[-1]['metrics']['pooled']['r2'])
 evidence=group_loo_affine_evidence(va['y'],np.mean(vp,0),va['groups'],upper=cap)
 approved=approve_dual_evidence([OutputCalibrator(run['chosen']) for run in runs],evidence)
 base=regression_metrics(te['y'],np.mean(tp,0),te['groups']);calibrated=regression_metrics(te['y'],np.mean(pred,0),te['groups'])
 result={'status':'fair calibration control; post-hoc','candidates':kinds,'runs':runs,
  'group_evidence':evidence,'dual_evidence_approved':approved,
  'base_ensemble':base,'calibrated_ensemble':calibrated,
  'final_ensemble':calibrated if approved else base}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 np.savez_compressed(OUT/'predictions.npz',validation_prediction=np.asarray(vp),validation_y=va['y'],validation_groups=va['groups'],base=np.asarray(tp),calibrated=np.asarray(pred),y=te['y'],groups=te['groups'])
 print('DONE',base['pooled']['r2'],calibrated['pooled']['r2'],approved)
if __name__=='__main__':main()
