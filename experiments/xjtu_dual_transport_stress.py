#!/usr/bin/env python3
"""Frozen dual-evidence output-transport stress test on XJTU condition shift."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from pp_extrapolation import (OutputCalibrator,approve_dual_evidence,approve_transport,fit_output_calibrator,
 fit_pp,group_loo_affine_evidence,predict,regression_metrics,
 select_affine_initialization,select_group_loo_calibrator,transport_direction_cosine)
from xjtu_untouched import CONDITIONS,build_cache,windows

ROOT=Path(__file__).resolve().parents[1];SEEDS=(42,43,44,45,46)

def main():
 torch.set_num_threads(2);raw=build_cache();train=windows(raw,'37.5Hz11kN');val=windows(raw,'35Hz12kN')
 test=windows(raw,'40Hz10kN');aff=select_affine_initialization(train,val);cap=float(aff['target_scale'])
 vp=[];tp=[];choices=[];calibrated=[]
 for seed in SEEDS:
  fit=fit_pp(train,val,seed=seed,affine_selection=aff,max_epochs=300)
  v=predict(fit,val['x']);t=predict(fit,test['x']);vp.append(v);tp.append(t)
  cal,scores=select_group_loo_calibrator(val['y'],v,val['groups'],upper=cap)
  calibrated.append(cal.predict(t,upper=cap));choices.append({'seed':seed,'kind':cal.kind,
   'slope':cal.slope,'intercept':cal.intercept,'loo_mse':scores})
 vp=np.asarray(vp);tp=np.asarray(tp);calibrated=np.asarray(calibrated)
 evidence=group_loo_affine_evidence(val['y'],vp.mean(0),val['groups'],upper=cap)
 calibrators=[OutputCalibrator(x['kind']) for x in choices]
 statistical_approval=approve_dual_evidence(calibrators,evidence)
 train_c=np.asarray([CONDITIONS['37.5Hz11kN']]);val_c=np.asarray([CONDITIONS['35Hz12kN']]);test_c=np.asarray([CONDITIONS['40Hz10kN']])
 cosine=transport_direction_cosine(train_c,val_c,test_c)
 approved=approve_transport(calibrators,evidence,direction_cosine=cosine)
 base=regression_metrics(test['y'],tp.mean(0),test['groups'])
 cal=regression_metrics(test['y'],calibrated.mean(0),test['groups'])
 result={'status':'post-hoc external stress test; XJTU labels were seen in earlier PP work',
  'shift_geometry':'validation and test are different operating-condition directions from train',
  'choices':choices,'group_evidence':evidence,'statistical_approval':statistical_approval,
  'direction_cosine':cosine,'transport_approved':approved,
  'base':base,'calibrated':cal,'final':cal if approved else base}
 out=ROOT/'results/xjtu_dual_transport_stress_v1';out.mkdir(parents=True,exist_ok=True)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print('choices',[x['kind'] for x in choices]);print('evidence',evidence['bootstrap_ci'])
 print('approved',approved,'base',base['pooled']['r2'],'calibrated',cal['pooled']['r2'],'final',result['final']['pooled']['r2'])

if __name__=='__main__':main()
