#!/usr/bin/env python3
"""Locked HNEI applicability-certificate evaluation."""
from __future__ import annotations
import json,pickle,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from calce_external_eval import rows
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,support_distance
from cross_domain_mechanism_study import descriptors,conformal,paired
ZIP=ROOT/'data/hnei_external/HNEI.zip';OUT=ROOT/'results/hnei_external_locked_v1';SEEDS=(42,43,44,45,46)

def load_series():
  out={}
  with zipfile.ZipFile(ZIP) as z:
    for name in sorted(n for n in z.namelist() if n.endswith('.pkl')):
      d=pickle.loads(z.read(name));cy=[];cap=[]
      for row in d['cycle_data']:
        v=np.asarray(row.get('discharge_capacity_in_Ah') or [],float);v=v[np.isfinite(v)]
        if len(v) and np.max(v)>0:cy.append(float(row['cycle_number']));cap.append(float(np.max(v)))
      u=np.unique(cy);vals=[float(np.median(np.asarray(cap)[np.asarray(cy)==q])) for q in u]
      out[d['cell_id']]={'cycle':u.tolist(),'capacity':vals,'nominal_capacity':d['nominal_capacity_in_Ah']}
  return out

def main():
  torch.set_num_threads(2);series=load_series();eligible=sorted(k for k,v in series.items() if len(v['cycle'])>=20);n=len(eligible);a=int(.6*n);b=a+int(.2*n);ids={'train':eligible[:a],'validation':eligible[a:b],'test':eligible[b:]}
  time_scale=max(np.asarray(series[u]['cycle'])[-1]-np.asarray(series[u]['cycle'])[0] for u in ids['train'])
  tr=rows(series,ids['train'],'train',time_scale);va=rows(series,ids['validation'],'eval',time_scale);te=rows(series,ids['test'],'eval',time_scale);desc=descriptors(tr,te)
  certificate=bool(desc['normalized_horizon']>.5 and desc['trajectory_heterogeneity']<.05 and desc['descriptor_units']>=4 and len(ids['validation'])>=2)
  aff=select_affine_initialization(tr,va);pv=[];pt=[];qv=[];qt=[];runs=[]
  for seed in SEEDS:
    fm=fit_plain(tr,va,seed=seed);fp=fit_pp(tr,va,seed=seed,affine_selection=aff);pv.append(predict_plain(fm,va['x']));pt.append(predict_plain(fm,te['x']));qv.append(predict(fp,va['x']));qt.append(predict(fp,te['x']))
    runs.append({'seed':seed,'plain_r2':regression_metrics(te['y'],pt[-1],te['groups'])['pooled']['r2'],'pp_r2':regression_metrics(te['y'],qt[-1],te['groups'])['pooled']['r2']});print(runs[-1],flush=True)
  pv,pt,qv,qt=map(np.asarray,(pv,pt,qv,qt));pve=pv.mean(0);qve=qv.mean(0);pe=pt.mean(0);qe=qt.mean(0);dv,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);dt,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]]);pm=regression_metrics(te['y'],pe,te['groups']);qm=regression_metrics(te['y'],qe,te['groups']);success=bool(qm['pooled']['r2']>0 and qm['pooled']['r2']>=pm['pooled']['r2']);vpm=regression_metrics(va['y'],pve,va['groups']);vqm=regression_metrics(va['y'],qve,va['groups']);selected='pp' if vqm['pooled']['rmse']<=vpm['pooled']['rmse'] else 'plain';selected_pred=qe if selected=='pp' else pe
  res={'status':'leakage-audited locked-split re-evaluation; developmental because normalization repair followed first outcome materialization','archive_md5':'b64ab2759d58350fc614602bd940159c','eligible_ids':eligible,'split_ids':ids,'n':{'train':len(tr['y']),'validation':len(va['y']),'test':len(te['y'])},'descriptors':desc,'predeclared_certificate_approved':certificate,'runs':runs,'plain_ensemble':pm,'pp_ensemble':qm,'pp_gain_r2':qm['pooled']['r2']-pm['pooled']['r2'],'primary_pp_success':success,'certificate_correct':bool(certificate==success),'validation_route':{'rule':'lower validation ensemble RMSE','plain':vpm,'pp':vqm,'selected':selected,'test_metrics':regression_metrics(te['y'],selected_pred,te['groups'])},'paired':paired(te['y'],te['groups'],pe,qe,np.random.default_rng(20260908)),'conformal90':conformal(va['y'],va['groups'],qve,te['y'],te['groups'],qe,dv,dt)}
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(res,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=te['y'],groups=te['groups'],plain=pt,pp=qt,test_distance=dt);print(json.dumps(res,indent=2))
if __name__=='__main__':main()
