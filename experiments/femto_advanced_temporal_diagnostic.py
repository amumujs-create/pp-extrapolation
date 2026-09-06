#!/usr/bin/env python3
"""Post-test diagnostic: does a causal GRU rescue FEMTO endpoint RUL?"""
import json,sys
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import regression_metrics
from pp_extrapolation.temporal import fit_temporal,predict_temporal

ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from femto_bearing_loader import FEATURE_COLS,load_femto_phm2012
SEEDS=(42,43,44,45,46);WINDOW=16
def sequence_part(df,units,endpoint_only=False):
 xs=[];ys=[];gs=[]
 for _,d in df[df.unit.isin(units)].groupby('bearing'):
  d=d.sort_values('cycle');x=d[FEATURE_COLS].to_numpy(np.float32);y=d.RUL.to_numpy(np.float32)
  indices=[len(d)-1] if endpoint_only else range(len(d))
  for i in indices:
   ix=np.maximum(np.arange(i-WINDOW+1,i+1),0);xs.append(x[ix]);ys.append(y[i]);gs.append(d.bearing.iloc[i])
 x=np.asarray(xs);n=len(x)
 return {'x':x,'y':np.asarray(ys,np.float32),'groups':np.asarray(gs),'prior':np.zeros((n,2),np.float32),'reliability':np.full((n,2),.5,np.float32)}
def main():
 torch.set_num_threads(2);df,g=load_femto_phm2012(root=Path('data/femto/raw'),file_stride=5)
 tr=sequence_part(df,g['train']);va=sequence_part(df,g['val']);te=sequence_part(df,g['test'],True);runs=[]
 for seed in SEEDS:
  fit=fit_temporal(tr,va,seed=seed,mode='direct',epochs=300);pred,_=predict_temporal(fit,te);m=regression_metrics(te['y'],pred,te['groups'])
  runs.append({'seed':seed,'selection':fit['selection'],'metrics':m});print(seed,fit['selection']['epoch'],m['pooled']['r2'],flush=True)
 r=np.asarray([x['metrics']['pooled']['r2'] for x in runs]);payload={'experiment':'femto_gru_post_test_diagnostic','status':'retrospective after PP test; not confirmatory','window':WINDOW,'seeds':SEEDS,'summary':{'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std())},'runs':runs}
 out=Path('results/femto_advanced_temporal_diagnostic_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
