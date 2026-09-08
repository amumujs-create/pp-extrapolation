#!/usr/bin/env python3
"""Prospective test-only Na-ion cohort with official 80%-capacity EOL."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from naion_prefix_gate_eval import _find,make_rows,prefix_descriptor
from naion_survival_projected_gate_v3_eval import row_ages,project
from naion_unit_route_gate_v4_eval import scaled_coordinates
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
OUT=ROOT/'results/naion_80eol_prospective_gate';SEEDS=(42,43,44,45,46)
TEST_NAMES=('270040-6-2-30.csv','270040-6-6-26.csv','270040-6-8-24.csv','270040-7-1-23.csv','270040-8-5-16.csv')

def read_file(p):
 f=pd.read_csv(p,encoding='gbk',low_memory=False);cc=_find(f.columns,'循环号');dc=_find(f.columns,'放电容量');z=pd.DataFrame({'cycle':pd.to_numeric(f[cc],errors='coerce'),'capacity':pd.to_numeric(f[dc].replace('-',np.nan),errors='coerce')}).dropna();z=z[(z.cycle>=0)&(z.capacity>0)];q=z.groupby('cycle',sort=True).capacity.max();return {'cycle':q.index.to_numpy(float),'capacity':q.to_numpy(float)}

def eol_truncate(z):
 hit=np.flatnonzero(z['capacity']<=.8)
 if not len(hit):return None
 i=int(hit[0]);span=z['cycle'][i]-z['cycle'][0]
 if span<40:return None
 return {'cycle':z['cycle'][:i+1],'capacity':z['capacity'][:i+1]}

def load_development():
 tr={};va={}
 for group in (1,2,3,4):
  directory=ROOT/'data'/('naion_external' if group==1 else f'naion_external_v{group}')
  for p in sorted(directory.glob(f'270040-{group}-*.csv')):
   z=eol_truncate(read_file(p))
   if z is not None:(tr if group<=3 else va)[p.name]=z
 return tr,va

def dist(train,val,test):
 zt,zall=scaled_coordinates(train,np.vstack([val,test]));zv=zall[:len(val)];zx=zall[len(val):];k=np.sqrt(train.shape[1]);return (np.linalg.norm(zx[:,None]-zt[None],axis=2)/k).min(1),(np.linalg.norm(zx[:,None]-zv[None],axis=2)/k).min(1)

def main():
 torch.set_num_threads(2);train,val=load_development();raw={n:read_file(ROOT/'data/naion_80eol_test'/n) for n in TEST_NAMES}
 tl={u:float(z['cycle'][-1]-z['cycle'][0]) for u,z in train.items()};vl={u:float(z['cycle'][-1]-z['cycle'][0]) for u,z in val.items()};boundary=float(np.floor(.6*np.median(list(tl.values()))));scale=max(tl.values())
 tr=make_rows(train,sorted(train),boundary,scale,'prefix');va=make_rows(val,sorted(val),boundary,scale,'tail');va_age=row_ages(val,sorted(val),boundary,'tail');aff=select_affine_initialization(tr,va);fm=[];fp=[];pv=[];qv=[]
 for seed in SEEDS:
  m=fit_plain(tr,va,seed=seed);n=fit_pp(tr,va,seed=seed,affine_selection=aff);fm.append(m);fp.append(n);pv.append(predict_plain(m,va['x']));qv.append(predict(n,va['x']))
 pv=project(np.asarray(pv),va_age,max(tl.values()));qv=project(np.asarray(qv),va_age,max(tl.values()));pvm=regression_metrics(va['y'],pv.mean(0),va['groups']);qvm=regression_metrics(va['y'],qv.mean(0),va['groups']);gain=1-qvm['pooled']['rmse']**2/max(pvm['pooled']['rmse']**2,1e-12);dis=float(np.mean(np.std(qv,0))/max(np.std(va['y']),1e-8))
 td=np.asarray([prefix_descriptor(train[u]['cycle'],train[u]['capacity'],boundary) for u in sorted(train)]);vd=np.asarray([prefix_descriptor(val[u]['cycle'],val[u]['capacity'],boundary) for u in sorted(val)]);available={u:z for u,z in raw.items() if z['cycle'][-1]-z['cycle'][0]>=boundary};xd=np.asarray([prefix_descriptor(available[u]['cycle'],available[u]['capacity'],boundary) for u in available]);dt,dv=dist(td,vd,xd)
 global_ok=bool(gain>=.02 and dis<=.25 and len(train)>=4 and len(val)>=2);decisions={u:{'train_distance':float(dt[i]),'validation_distance':float(dv[i]),'approved':bool(global_ok and dt[i]<=3 and dv[i]<=1.5)} for i,u in enumerate(available)}
 gate={'protocol':'protocols/NAION_80EOL_PROSPECTIVE_GATE_PROTOCOL.md','train_ids':sorted(train),'validation_ids':sorted(val),'fixed_test_ids':TEST_NAMES,'boundary':boundary,'validation':{'plain':pvm,'pp':qvm,'relative_mse_gain':gain,'seed_disagreement':dis},'global_approved':global_ok,'unit_decisions':decisions,'prefix_unavailable_before_boundary':sorted(set(TEST_NAMES)-set(available)),'test_prefix_sha256':hashlib.sha256(json.dumps({u:xd[i].tolist() for i,u in enumerate(available)},sort_keys=True).encode()).hexdigest()}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'gate_decision_preoutcome.json').write_text(json.dumps(gate,indent=2)+'\n')
 test={};censored=[]
 for u,z in raw.items():
  q=eol_truncate(z)
  if q is None:censored.append(u)
  elif u in available:test[u]=q
 if not test:raise RuntimeError('no evaluable failure-observed test cells')
 te=make_rows(test,sorted(test),boundary,scale,'tail');age=row_ages(test,sorted(test),boundary,'tail');known=max(list(tl.values())+list(vl.values()));pr=np.asarray([predict_plain(m,te['x']) for m in fm]);qr=np.asarray([predict(m,te['x']) for m in fp]);pp=project(pr,age,known);qp=project(qr,age,known);pe=pp.mean(0);qe=qp.mean(0);route=pe.copy();per={};correct=[];approved=[]
 for u in sorted(test):
  m=te['groups']==u;pm=regression_metrics(te['y'][m],pe[m],te['groups'][m]);qm=regression_metrics(te['y'][m],qe[m],te['groups'][m]);actual=bool(qm['pooled']['r2']>0 and qm['pooled']['rmse']<=pm['pooled']['rmse']);decision=decisions[u]['approved'];route[m]=qe[m] if decision else pe[m];correct.append(decision==actual);approved.extend(np.flatnonzero(m).tolist() if decision else []);per[u]={'decision':'PP' if decision else 'MLP','actual_pp_success':actual,'correct':decision==actual,'plain':pm,'pp':qm}
 approved=np.asarray(approved,int);plain_all=regression_metrics(te['y'],pe,te['groups']);pp_all=regression_metrics(te['y'],qe,te['groups']);deployment=regression_metrics(te['y'],route,te['groups']);selective=None if not len(approved) else regression_metrics(te['y'][approved],qe[approved],te['groups'][approved]);coverage=len(approved)/len(te['y'])
 success=bool(coverage>0 and all(correct) and selective['pooled']['r2']>0 and deployment['pooled']['r2']>=plain_all['pooled']['r2'])
 result={'status':'prospective one-shot test-only external cohort','gate':gate,'right_censored_test_ids':censored,'evaluable_test_ids':sorted(test),'n':{'train_units':len(train),'validation_units':len(val),'test_units':len(test),'test_rows':len(te['y'])},'projected_all':{'plain':plain_all,'pp':pp_all},'per_unit_route':per,'row_coverage':coverage,'unit_route_accuracy':float(np.mean(correct)),'selective_pp':selective,'deployment_route':deployment,'prospective_gate_success':success}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=te['y'],groups=te['groups'],plain=pp,pp=qp,route=route,age=age);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
