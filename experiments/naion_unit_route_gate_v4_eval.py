#!/usr/bin/env python3
"""Prospective unit-level PP routing on untouched Na-ion group 4."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from naion_prefix_gate_eval import _find,make_rows,prefix_descriptor
from naion_survival_projected_gate_v3_eval import row_ages,project
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization

DATA=ROOT/'data/naion_external_v4';OUT=ROOT/'results/naion_unit_route_gate_v4';SEEDS=(42,43,44,45,46)
EXPECTED=('270040-4-1-48.csv','270040-4-2-47.csv','270040-4-3-46.csv','270040-4-4-45.csv',
          '270040-4-5-44.csv','270040-4-6-43.csv','270040-4-7-42.csv','270040-4-8-41.csv')

def load_series():
  out={}
  for name in EXPECTED:
    p=DATA/name;f=pd.read_csv(p,encoding='gbk');cc=_find(f.columns,'循环号');dc=_find(f.columns,'放电容量')
    z=pd.DataFrame({'cycle':pd.to_numeric(f[cc],errors='coerce'),'capacity':pd.to_numeric(f[dc].replace('-',np.nan),errors='coerce')}).dropna()
    z=z[(z.cycle>=0)&(z.capacity>0)];q=z.groupby('cycle',sort=True).capacity.max();t,c=q.index.to_numpy(float),q.to_numpy(float)
    if len(t)>=40:out[name]={'cycle':t,'capacity':c}
  return out

def scaled_coordinates(train,other):
  center=np.median(train,0);q25,q75=np.quantile(train,[.25,.75],axis=0)
  scale=np.maximum(q75-q25,np.maximum(np.abs(center)*.05,1e-4))
  return (train-center)/scale,(other-center)/scale

def main():
  torch.set_num_threads(2);s=load_series();all_ids=sorted(s)
  if tuple(all_ids)!=EXPECTED:raise RuntimeError(f'locked cohort mismatch: {all_ids}')
  ids={'train':all_ids[:4],'validation':all_ids[4:6],'test':all_ids[6:]}
  life={u:float(s[u]['cycle'][-1]-s[u]['cycle'][0]) for u in all_ids};tl=[life[u] for u in ids['train']]
  boundary=float(np.floor(.6*np.median(tl)));scale=float(max(tl));tr=make_rows(s,ids['train'],boundary,scale,'prefix');va=make_rows(s,ids['validation'],boundary,scale,'tail');va_age=row_ages(s,ids['validation'],boundary,'tail')
  aff=select_affine_initialization(tr,va);fm=[];fp=[];pv=[];qv=[]
  for seed in SEEDS:
    m=fit_plain(tr,va,seed=seed);n=fit_pp(tr,va,seed=seed,affine_selection=aff);fm.append(m);fp.append(n);pv.append(predict_plain(m,va['x']));qv.append(predict(n,va['x']))
  pv=project(np.asarray(pv),va_age,max(tl));qv=project(np.asarray(qv),va_age,max(tl));pvm=regression_metrics(va['y'],pv.mean(0),va['groups']);qvm=regression_metrics(va['y'],qv.mean(0),va['groups'])
  gain=1-qvm['pooled']['rmse']**2/max(pvm['pooled']['rmse']**2,1e-12);dis=float(np.mean(np.std(qv,0))/max(np.std(va['y']),1e-8))
  td=np.asarray([prefix_descriptor(s[u]['cycle'],s[u]['capacity'],boundary) for u in ids['train']]);vd=np.asarray([prefix_descriptor(s[u]['cycle'],s[u]['capacity'],boundary) for u in ids['validation']]);xd=np.asarray([prefix_descriptor(s[u]['cycle'],s[u]['capacity'],boundary) for u in ids['test']])
  zt,zall=scaled_coordinates(td,np.vstack([vd,xd]));zv,zx=zall[:2],zall[2:]
  dt=(np.linalg.norm(zx[:,None]-zt[None],axis=2)/np.sqrt(td.shape[1])).min(1)
  dval=(np.linalg.norm(zx[:,None]-zv[None],axis=2)/np.sqrt(td.shape[1])).min(1)
  global_checks={'validation_mse_gain_at_least_2pct':bool(gain>=.02),'seed_disagreement_at_most_0_25':bool(dis<=.25),'enough_units':bool(len(ids['train'])>=4 and len(ids['validation'])>=2)}
  unit_checks={u:{'train_distance':float(dt[i]),'validation_distance':float(dval[i]),'approved':bool(all(global_checks.values()) and dt[i]<=3 and dval[i]<=1.5)} for i,u in enumerate(ids['test'])}
  gate={'protocol':'protocols/NAION_UNIT_ROUTE_GATE_V4_PROTOCOL.md','split_ids':ids,'boundary_train_cycle_span':boundary,'known_max_life_validation':max(tl),'validation_projected':{'plain':pvm,'pp':qvm,'relative_mse_gain':gain,'normalized_seed_disagreement':dis},'global_checks':global_checks,'unit_decisions':unit_checks,'coverage':float(np.mean([x['approved'] for x in unit_checks.values()])),'test_prefix_commitment_sha256':hashlib.sha256(json.dumps({'test_ids':ids['test'],'test_prefix_descriptors':xd.tolist()},sort_keys=True).encode()).hexdigest()}
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'gate_decision_preoutcome.json').write_text(json.dumps(gate,indent=2)+'\n')
  te=make_rows(s,ids['test'],boundary,scale,'tail');age=row_ages(s,ids['test'],boundary,'tail');known=max(life[u] for u in ids['train']+ids['validation']);pr=np.asarray([predict_plain(m,te['x']) for m in fm]);qr=np.asarray([predict(m,te['x']) for m in fp]);pp=project(pr,age,known);qp=project(qr,age,known);pe=pp.mean(0);qe=qp.mean(0);route=pe.copy()
  per={};correct=[];approved=[]
  for u in ids['test']:
    m=te['groups']==u;pm=regression_metrics(te['y'][m],pe[m],te['groups'][m]);qm=regression_metrics(te['y'][m],qe[m],te['groups'][m]);actual=bool(qm['pooled']['r2']>0 and qm['pooled']['rmse']<=pm['pooled']['rmse']);decision=unit_checks[u]['approved'];route[m]=qe[m] if decision else pe[m];correct.append(decision==actual);approved.extend(np.flatnonzero(m).tolist() if decision else []);per[u]={'decision':'PP' if decision else 'MLP','actual_pp_success':actual,'correct':decision==actual,'plain':pm,'pp':qm}
  approved=np.asarray(approved,int);sel=None if not len(approved) else regression_metrics(te['y'][approved],qe[approved],te['groups'][approved])
  result={'status':'prospective one-shot external confirmation','development_source':'group-3 mixed unit response; no group-4 outcome used','gate':gate,'n':{'train':len(tr['y']),'validation':len(va['y']),'test':len(te['y'])},'projected_all':{'plain':regression_metrics(te['y'],pe,te['groups']),'pp':regression_metrics(te['y'],qe,te['groups'])},'per_unit_route':per,'unit_route_accuracy':float(np.mean(correct)),'selective_pp_metrics':sel,'deployment_route_metrics':regression_metrics(te['y'],route,te['groups']),'gate_success':bool(all(correct) and gate['coverage']>0)}
  (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=te['y'],groups=te['groups'],plain=pp,pp=qp,route=route,age=age);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
