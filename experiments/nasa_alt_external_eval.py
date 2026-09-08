#!/usr/bin/env python3
"""Locked external evaluation on NASA/UCF randomized ALT batteries."""
from __future__ import annotations
import io,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,support_distance
from cross_domain_mechanism_study import descriptors,conformal,paired
ZIP=ROOT/'data/nasa_alt_external/battery_alt_dataset.zip';OUT=ROOT/'results/nasa_alt_external_locked_v1';SEEDS=(42,43,44,45,46)

def parse_one(z,name):
    episodes=[];cap=dur=0.;last_t=last_i=None;first_t=None;row0=0
    with z.open(name) as raw:
      for chunk in pd.read_csv(raw,usecols=['time','mode','current_load','mission_type'],chunksize=500000):
        t=pd.to_numeric(chunk.time,errors='coerce').to_numpy();mode=pd.to_numeric(chunk['mode'],errors='coerce').to_numpy();cur=pd.to_numeric(chunk.current_load,errors='coerce').to_numpy();mission=pd.to_numeric(chunk.mission_type,errors='coerce').to_numpy()
        idx=np.flatnonzero((mode<-.5)&(mission<.5)&np.isfinite(t)&np.isfinite(cur))
        for j in idx:
          gi=row0+j;tt=float(t[j]);cc=abs(float(cur[j]))
          contiguous=last_i is not None and gi==last_i+1 and 0<tt-last_t<=10
          if not contiguous:
            if cap>.1 and dur>60:episodes.append({'time':first_t,'capacity_ah':cap,'duration_s':dur})
            cap=dur=0.;first_t=tt
          else:
            dt=tt-last_t;cap+=(last_c+cc)*.5*dt/3600;dur+=dt
          last_t,last_c,last_i=tt,cc,gi
        row0+=len(chunk)
    if cap>.1 and dur>60:episodes.append({'time':first_t,'capacity_ah':cap,'duration_s':dur})
    # Reference discharges cluster near 2.5A and should yield plausible pack capacity.
    episodes=[e for e in episodes if .2<=e['capacity_ah']<=10]
    return episodes

def extract():
    cache=OUT/'capacity_trajectories.json'
    if cache.exists():return json.load(open(cache))
    OUT.mkdir(parents=True,exist_ok=True);data={}
    with zipfile.ZipFile(ZIP) as z:
      names=sorted(n for n in z.namelist() if 'regular_alt_batteries/battery' in n and n.endswith('.csv'))
      for name in names:
        key=Path(name).stem;data[key]=parse_one(z,name);print('PARSED',key,len(data[key]),flush=True)
    cache.write_text(json.dumps(data,indent=2)+'\n');return data

def feature_rows(series,ids,part,time_scale):
    rows=[];ys=[];groups=[]
    for uid in ids:
      e=series[uid];t=np.array([q['time'] for q in e],float);c=np.array([q['capacity_ah'] for q in e],float)
      if len(c)<10:continue
      h=c/max(c[0],1e-8);life=t[-1]-t[0];progress=(t-t[0])/max(life,1e-8);elapsed=(t-t[0])/max(time_scale,1e-8);rul=(t[-1]-t)/86400
      start=3;mask=(progress<=.70) if part=='train' else (progress>.70)
      for i in np.flatnonzero(mask & (np.arange(len(c))>=start)):
        slopes=[(h[i]-h[max(0,i-w)])/(elapsed[i]-elapsed[max(0,i-w)]+1e-8) for w in (1,2,3)]
        hist=h[max(0,i-3):i+1]
        x=[h[i],elapsed[i],*slopes,float(hist.mean()),float(hist.std()),float(h[i]-h[0]),float(np.min(h[:i+1])),float((t[i]-t[0])/max(time_scale,1e-8)),1.0]
        rows.append(x);ys.append(rul[i]);groups.append(uid)
    return {'x':np.asarray(rows,np.float32),'y':np.asarray(ys,np.float32),'groups':np.asarray(groups)}

def main():
    torch.set_num_threads(2);series=extract();eligible=sorted(k for k,v in series.items() if len(v)>=10);n=len(eligible);a=int(.6*n);b=a+int(.2*n);ids={'train':eligible[:a],'validation':eligible[a:b],'test':eligible[b:]}
    time_scale=max(series[u][-1]['time']-series[u][0]['time'] for u in ids['train'])
    tr=feature_rows(series,ids['train'],'train',time_scale);va=feature_rows(series,ids['validation'],'eval',time_scale);te=feature_rows(series,ids['test'],'eval',time_scale)
    if min(len(tr['y']),len(va['y']),len(te['y']))==0:raise RuntimeError('locked split infeasible after predeclared eligibility')
    aff=select_affine_initialization(tr,va);pv=[];pt=[];qv=[];qt=[];runs=[]
    for seed in SEEDS:
      fm=fit_plain(tr,va,seed=seed);fp=fit_pp(tr,va,seed=seed,affine_selection=aff)
      pv.append(predict_plain(fm,va['x']));pt.append(predict_plain(fm,te['x']));qv.append(predict(fp,va['x']));qt.append(predict(fp,te['x']))
      runs.append({'seed':seed,'plain_r2':regression_metrics(te['y'],pt[-1],te['groups'])['pooled']['r2'],'pp_r2':regression_metrics(te['y'],qt[-1],te['groups'])['pooled']['r2']});print(runs[-1],flush=True)
    pv,pt,qv,qt=map(np.asarray,(pv,pt,qv,qt));pve=pv.mean(0);qve=qv.mean(0);pe=pt.mean(0);qe=qt.mean(0);dv,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);dt,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
    val_plain=regression_metrics(va['y'],pve,va['groups']);val_pp=regression_metrics(va['y'],qve,va['groups'])
    selected='pp' if val_pp['pooled']['rmse']<=val_plain['pooled']['rmse'] else 'plain';selected_pred=qe if selected=='pp' else pe
    result={'status':'leakage-audited locked-split re-evaluation; developmental because normalization repair followed first outcome materialization','archive_sha256':'89209acddbad47506d781698fc51ebe9d7cdb8968667698f63e63aa730981d21','eligible_ids':eligible,'split_ids':ids,'n':{'train_rows':len(tr['y']),'validation_rows':len(va['y']),'test_rows':len(te['y'])},'descriptors':descriptors(tr,te),'runs':runs,
      'plain_ensemble':regression_metrics(te['y'],pe,te['groups']),'pp_ensemble':regression_metrics(te['y'],qe,te['groups']),'pp_gain_r2':float(regression_metrics(te['y'],qe,te['groups'])['pooled']['r2']-regression_metrics(te['y'],pe,te['groups'])['pooled']['r2']),
      'validation_route':{'rule':'lower validation ensemble RMSE','plain':val_plain,'pp':val_pp,'selected':selected,'test_metrics':regression_metrics(te['y'],selected_pred,te['groups'])},
      'paired':paired(te['y'],te['groups'],pe,qe,np.random.default_rng(20260908)),'conformal90':conformal(va['y'],va['groups'],qve,te['y'],te['groups'],qe,dv,dt)}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=te['y'],groups=te['groups'],plain=pt,pp=qt,test_distance=dt)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
