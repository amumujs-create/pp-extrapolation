#!/usr/bin/env python3
"""Locked CALCE CS2/CX2 external evaluation from BatteryLife Zenodo release."""
from __future__ import annotations
import json,pickle,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,support_distance
from cross_domain_mechanism_study import descriptors,conformal,paired
ZIP=ROOT/'data/calce_external/CALCE.zip';OUT=ROOT/'results/calce_external_locked_v1';SEEDS=(42,43,44,45,46)

def load_series():
  out={}
  with zipfile.ZipFile(ZIP) as z:
    for name in sorted(n for n in z.namelist() if n.endswith('.pkl')):
      d=pickle.loads(z.read(name));cy=[];cap=[]
      for row in d['cycle_data']:
        v=np.asarray(row.get('discharge_capacity_in_Ah') or [],float);v=v[np.isfinite(v)]
        if len(v) and np.max(v)>0:cy.append(float(row['cycle_number']));cap.append(float(np.max(v)))
      # Collapse duplicate cycle numbers deterministically by median.
      u=np.unique(cy);vals=[float(np.median(np.asarray(cap)[np.asarray(cy)==q])) for q in u]
      out[d['cell_id']]={'cycle':u.tolist(),'capacity':vals,'nominal_capacity':d['nominal_capacity_in_Ah']}
  return out

def rows(series,ids,part,time_scale):
  xx=[];yy=[];gg=[]
  for uid in ids:
    e=series[uid];t=np.asarray(e['cycle'],float);c=np.asarray(e['capacity'],float);h=c/max(c[0],1e-8);progress=(t-t[0])/max(t[-1]-t[0],1e-8);elapsed=(t-t[0])/max(time_scale,1e-8);rul=t[-1]-t
    mask=(progress<=.7) if part=='train' else (progress>.7)
    for i in np.flatnonzero(mask & (np.arange(len(t))>=3)):
      slopes=[(h[i]-h[max(0,i-w)])/(elapsed[i]-elapsed[max(0,i-w)]+1e-8) for w in (1,2,3)];hist=h[max(0,i-3):i+1]
      xx.append([h[i],elapsed[i],*slopes,float(hist.mean()),float(hist.std()),float(h[i]-h[0]),float(np.min(h[:i+1])),float((t[i]-t[0])/max(time_scale,1e-8)),1.]);yy.append(rul[i]);gg.append(uid)
  return {'x':np.asarray(xx,np.float32),'y':np.asarray(yy,np.float32),'groups':np.asarray(gg)}

def main():
  torch.set_num_threads(2);series=load_series();eligible=sorted(k for k,v in series.items() if len(v['cycle'])>=20);n=len(eligible);a=int(.6*n);b=a+int(.2*n);ids={'train':eligible[:a],'validation':eligible[a:b],'test':eligible[b:]}
  time_scale=max(np.asarray(series[u]['cycle'])[-1]-np.asarray(series[u]['cycle'])[0] for u in ids['train'])
  tr=rows(series,ids['train'],'train',time_scale);va=rows(series,ids['validation'],'eval',time_scale);te=rows(series,ids['test'],'eval',time_scale);aff=select_affine_initialization(tr,va);pv=[];pt=[];qv=[];qt=[];runs=[]
  for seed in SEEDS:
    fm=fit_plain(tr,va,seed=seed);fp=fit_pp(tr,va,seed=seed,affine_selection=aff);pv.append(predict_plain(fm,va['x']));pt.append(predict_plain(fm,te['x']));qv.append(predict(fp,va['x']));qt.append(predict(fp,te['x']))
    runs.append({'seed':seed,'plain_r2':regression_metrics(te['y'],pt[-1],te['groups'])['pooled']['r2'],'pp_r2':regression_metrics(te['y'],qt[-1],te['groups'])['pooled']['r2']});print(runs[-1],flush=True)
  pv,pt,qv,qt=map(np.asarray,(pv,pt,qv,qt));pve=pv.mean(0);qve=qv.mean(0);pe=pt.mean(0);qe=qt.mean(0);dv,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);dt,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
  pm=regression_metrics(te['y'],pe,te['groups']);qm=regression_metrics(te['y'],qe,te['groups']);vpm=regression_metrics(va['y'],pve,va['groups']);vqm=regression_metrics(va['y'],qve,va['groups']);selected='pp' if vqm['pooled']['rmse']<=vpm['pooled']['rmse'] else 'plain';selected_pred=qe if selected=='pp' else pe
  res={'status':'leakage-audited locked-split re-evaluation; developmental because normalization repair followed first outcome materialization','archive_md5':'fa67e99c93ac2795328da34cd64d9118','eligible_ids':eligible,'split_ids':ids,'n':{'train':len(tr['y']),'validation':len(va['y']),'test':len(te['y'])},'descriptors':descriptors(tr,te),'runs':runs,'plain_ensemble':pm,'pp_ensemble':qm,'pp_gain_r2':qm['pooled']['r2']-pm['pooled']['r2'],'validation_route':{'rule':'lower validation ensemble RMSE','plain':vpm,'pp':vqm,'selected':selected,'test_metrics':regression_metrics(te['y'],selected_pred,te['groups'])},'paired':paired(te['y'],te['groups'],pe,qe,np.random.default_rng(20260908)),'conformal90':conformal(va['y'],va['groups'],qve,te['y'],te['groups'],qe,dv,dt)}
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(res,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=te['y'],groups=te['groups'],plain=pt,pp=qt,test_distance=dt);print(json.dumps(res,indent=2))
if __name__=='__main__':main()
