#!/usr/bin/env python3
"""Locked PP model-level prospective evaluation on PHM2012 FEMTO endpoints."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import audit_convex_hull_support,fit_pp,predict,regression_metrics,select_affine_initialization
from pp_extrapolation.model import _affine_prediction
from plain_mlp_ablation import fit_plain,predict_plain

ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from femto_bearing_loader import FEATURE_COLS,load_femto_phm2012
SEEDS=(42,43,44,45,46)
def split(frame,units):
 d=frame[frame.unit.isin(units)].copy();return {'x':d[FEATURE_COLS].to_numpy(np.float32),'y':d.RUL.to_numpy(np.float32),'groups':d.bearing.to_numpy()}
def summary(y,m,g):
 scores=[regression_metrics(y,p,g)['pooled'] for p in m];r=np.array([x['r2'] for x in scores]);return {'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std()),'ensemble':regression_metrics(y,m.mean(0),g),'per_seed':scores}
def main():
 torch.set_num_threads(2);start=time.time();df,groups=load_femto_phm2012(root=Path('data/femto/raw'),file_stride=5)
 train=split(df,groups['train']);val=split(df,groups['val'])
 endpoint=df[df.unit.isin(groups['test'])].sort_values('cycle').groupby('unit',as_index=False).tail(1)
 test={'x':endpoint[FEATURE_COLS].to_numpy(np.float32),'y':endpoint.RUL.to_numpy(np.float32),'groups':endpoint.bearing.to_numpy()}
 hull=audit_convex_hull_support(train['x'],test['x'],feature_names=FEATURE_COLS)
 affine=select_affine_initialization(train,val);ridge=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale'])
 pp=[];plain=[];runs=[]
 for seed in SEEDS:
  pf=fit_pp(train,val,seed=seed,affine_selection=affine,max_epochs=300);nf=fit_plain(train,val,seed=seed,max_epochs=300)
  p=predict(pf,test['x']);n=predict_plain(nf,test['x']);pp.append(p);plain.append(n)
  runs.append({'seed':seed,'pp':regression_metrics(test['y'],p,test['groups']),'plain_nn':regression_metrics(test['y'],n,test['groups'])})
  print(seed,'PP',runs[-1]['pp']['pooled']['r2'],'plain',runs[-1]['plain_nn']['pooled']['r2'],flush=True)
 pp=np.asarray(pp);plain=np.asarray(plain)
 payload={'experiment':'femto_pp_model_level_prospective_v1','protocol_commit':'a44b528','evidence_label':'PP model-level prospective; PAE previously inspected dataset',
 'n':{'train_rows':len(train['y']),'validation_rows':len(val['y']),'test_bearings':len(test['y'])},'units':groups,
 'test_hull':hull.summary(),'ridge':regression_metrics(test['y'],ridge,test['groups']),'pp':summary(test['y'],pp,test['groups']),'plain_nn':summary(test['y'],plain,test['groups']),
 'test_truth':test['y'].tolist(),'test_bearings':test['groups'].tolist(),'runs':runs,'runtime_seconds':time.time()-start}
 out=Path('results/femto_pp_prospective_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps({k:payload[k] for k in ('n','test_hull','ridge','pp','plain_nn')},indent=2,default=str))
if __name__=='__main__':main()
