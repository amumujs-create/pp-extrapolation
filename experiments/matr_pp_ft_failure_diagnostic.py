#!/usr/bin/env python3
"""Row/cell-level diagnosis of latent PP versus FT on MATR2019 strict tail."""
from __future__ import annotations
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from matr_2019_latent_confirmatory import load_cells,make_rows
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,select_affine_initialization,regression_metrics,support_distance

OUT=ROOT/'results/matr_pp_ft_failure_diagnostic_v1'

def matlab_text(handle,ref):
 import h5py
 a=np.asarray(handle[ref]).reshape(-1);return ''.join(chr(int(v)) for v in a if int(v))

def metadata():
 import h5py
 from matr_2019_latent_confirmatory import MAT
 out={}
 with h5py.File(MAT,'r') as f:
  for i in range(f['batch/summary'].shape[0]):
   try:
    s=f[f['batch/summary'][i,0]];q=np.asarray(s['QDischarge']).reshape(-1);cy=np.asarray(s['cycle']).reshape(-1)
    if len(q)!=len(cy):continue
    out[f'c{i}']={'policy':matlab_text(f,f['batch/policy_readable'][i,0]),'life':float(cy[-1]),
                  'q0':float(np.median(q[:8])),'qend':float(q[-1])}
   except Exception:pass
 return out

def binned(y,predictions,values,bins):
 labels=np.digitize(values,bins[1:-1],right=True);result=[]
 for i in range(len(bins)-1):
  m=labels==i
  result.append({'lo':float(bins[i]),'hi':float(bins[i+1]),'n':int(m.sum()),
                 **{name:regression_metrics(y[m],pred[m],np.arange(m.sum()))['pooled'] for name,pred in predictions.items()}})
 return result

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 source=json.load(open(ROOT/'results/matr_2019_latent_confirmatory/results.json'));audit=source['pretest'];cells={c['index']:c for c in load_cells()[0]}
 parts=[make_rows([cells[i] for i in audit['split_indices'][name]],audit['actual_boundary'],train=name=='train') for name in ('train','validation','test')]
 tr,va,te=parts;aff=select_affine_initialization(tr,va);cache=OUT/'latent_predictions.npz'
 if cache.exists():
  z=np.load(cache);pp=np.asarray(z['prediction']);gates=np.asarray(z['gates'])
 else:
  pp=[];gates=[]
  for run in source['runs']['latent']:
   s=run['selection'];fit=fit_latent_regime_pp(tr,va,seed=s['seed'],affine_selection=aff,max_epochs=300,patience=70,
       separation_weight=run['separation'],gate_weight=run['gate_weight'])
   p,g=predict_latent_regime(fit,te['x'],return_gate=True);pp.append(p);gates.append(g);print('PP',s['seed'],flush=True)
  pp=np.asarray(pp);gates=np.asarray(gates);np.savez_compressed(cache,prediction=pp,gates=gates,y=te['y'],groups=te['groups'])
 ft=[]
 for seed in range(42,47):ft.append(np.load(ROOT/f'results/ft_original_budget_v1/matr2019/seed{seed}.npz')['prediction'])
 ft=np.asarray(ft);ppe=pp.mean(0);fte=ft.mean(0);y=te['y'].astype(float);groups=te['groups'];meta=metadata()
 distance,_=support_distance(tr['coordinate'][:,None],te['coordinate'][:,None]);distance=np.asarray(distance)
 train_policies={meta[f'c{i}']['policy'] for i in audit['split_indices']['train']};val_policies={meta[f'c{i}']['policy'] for i in audit['split_indices']['validation']}
 per=[]
 for group in np.unique(groups):
  m=groups==group;pm=regression_metrics(y[m],ppe[m],groups[m])['pooled'];fm=regression_metrics(y[m],fte[m],groups[m])['pooled'];d=meta[group]
  per.append({'unit':group,'n':int(m.sum()),**d,'policy_seen_train':d['policy'] in train_policies,'policy_seen_validation':d['policy'] in val_policies,
              'target_mean':float(y[m].mean()),'target_sd':float(y[m].std()),'distance_median':float(np.median(distance[m])),
              'pp_r2':pm['r2'],'pp_rmse':pm['rmse'],'pp_bias':float((ppe[m]-y[m]).mean()),
              'ft_r2':fm['r2'],'ft_rmse':fm['rmse'],'ft_bias':float((fte[m]-y[m]).mean()),'winner':'PP' if pm['rmse']<fm['rmse'] else 'FT'})
 def calibration(pred):
  slope,intercept=np.polyfit(pred,y,1);return {'prediction_mean':float(pred.mean()),'target_mean':float(y.mean()),'bias':float((pred-y).mean()),
      'correlation':float(np.corrcoef(pred,y)[0,1]),'calibration_y_on_prediction_slope':float(slope),'intercept':float(intercept)}
 pred={'pp':ppe,'ft':fte};yq=np.quantile(y,[0,.25,.5,.75,1]);dq=np.quantile(distance,[0,1/3,2/3,1])
 result={'status':'post-hoc failure diagnosis; no model selection','metrics':{k:regression_metrics(y,v,groups) for k,v in pred.items()},
  'single_seed':{k:{'mean_r2':float(np.mean([regression_metrics(y,p,groups)['pooled']['r2'] for p in a])),
                    'sd_r2':float(np.std([regression_metrics(y,p,groups)['pooled']['r2'] for p in a],ddof=1))} for k,a in [('pp',pp),('ft',ft)]},
  'calibration':{k:calibration(v) for k,v in pred.items()},'gate':{'mean':float(gates.mean()),'sd':float(gates.std()),
      'min':float(gates.min()),'max':float(gates.max()),'fraction_gt_0_99':float((gates>.99).mean())},
  'policy_counts':{'train':dict(Counter(meta[f'c{i}']['policy'] for i in audit['split_indices']['train'])),
                   'validation':dict(Counter(meta[f'c{i}']['policy'] for i in audit['split_indices']['validation'])),
                   'test':dict(Counter(meta[f'c{i}']['policy'] for i in audit['split_indices']['test']))},
  'rul_quartiles':binned(y,pred,y,yq),'distance_tertiles':binned(y,pred,distance,dq),'per_unit':per}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',result['metrics']['pp']['pooled']['r2'],result['metrics']['ft']['pooled']['r2'])
if __name__=='__main__':main()
