#!/usr/bin/env python3
"""Near/mid/far strict-hull shell statistics for frozen PP predictions."""
import json,sys
from pathlib import Path
import numpy as np
from pp_extrapolation import regression_metrics,support_distance
ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from run_affine_tail_external_three import prepare_hust,prepare_virkler
from run_affine_tail_external_nasa_health_v2 import prepare_folds
BASE=Path('results/support_gated_cross_domain_v1')
def matrix(a,prefix):return np.asarray([a[k] for k in a.files if k.startswith(prefix)])
def shell_metrics(y,g,p):
 values=[regression_metrics(y,x,g)['pooled'] for x in p];r=np.array([np.nan if x['r2'] is None else x['r2'] for x in values]);rm=np.array([x['rmse'] for x in values]);return {'r2_mean':None if np.isnan(r).all() else float(np.nanmean(r)),'r2_sd':None if np.isnan(r).all() else float(np.nanstd(r)),'rmse_mean':float(rm.mean()),'ensemble':regression_metrics(y,p.mean(0),g)['pooled']}
def run(name,tr,te,npz,dist=None):
 a=np.load(npz,allow_pickle=True);y=a['truth'];g=a['groups'];d=dist if dist is not None else support_distance(tr['x'][:,[0]],te['x'][:,[0]])[0]
 order=np.argsort(d,kind='stable');labels=np.empty(len(d),dtype=object)
 for label,index in zip(('near','mid','far'),np.array_split(order,3)):labels[index]=label
 out={}
 for label in ('near','mid','far'):
  m=labels==label;out[label]={'n':int(m.sum()),'distance':{'min':float(d[m].min()),'max':float(d[m].max()),'median':float(np.median(d[m]))},
   'affine':shell_metrics(y[m],g[m],matrix(a,'affine_seed')[:,m]),'pp':shell_metrics(y[m],g[m],matrix(a,'pp_seed')[:,m]),'support_pp':shell_metrics(y[m],g[m],matrix(a,'gated_seed')[:,m])}
 return out
def main():
 ht,_,he,_=prepare_hust();vt,_,ve,_=prepare_virkler();folds,_=prepare_folds();nd=np.concatenate([support_distance(f['train']['x'][:,[0]],f['test']['x'][:,[0]])[0] for f in folds])
 payload={'experiment':'distance_shell_statistics_v1','primary_interpretation':'RMSE; shell R2 can be unstable under restricted target range','datasets':{
  'hust':run('hust',ht,he,BASE/'predictions_hust.npz'),'virkler':run('virkler',vt,ve,BASE/'predictions_virkler.npz'),
  'nasa':run('nasa',folds[0]['train'],folds[0]['test'],BASE/'predictions_nasa.npz',nd)}}
 out=Path('results/distance_shell_statistics_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 for n,d in payload['datasets'].items():
  print(n,[(s,round(x['affine']['ensemble']['rmse'],3),round(x['pp']['ensemble']['rmse'],3),round(x['support_pp']['ensemble']['rmse'],3)) for s,x in d.items()])
if __name__=='__main__':main()
