#!/usr/bin/env python3
"""Matched PP/MLP, train-only mechanisms, paired inference, and block conformal."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import median_abs_deviation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from extrapolation_competitors_all import datasets as generic_datasets
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,support_distance

SEEDS=(42,43,44,45,46)

def all_datasets():
    out=generic_datasets()
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    out['nasa_battery']={'folds':prepare_folds()[0]}
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_tra_quantile_split import make_tra_hard_split
    from ncmapss_pp_benchmark import rows
    s=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42)
    names=list(FEATURE_COLS);out['ncmapss']=(rows(s.train,names),rows(s.val,names),rows(s.test,names))
    return out

def r2(y,p):
    den=np.sum((y-y.mean())**2);return float(1-np.sum((y-p)**2)/den) if den>0 else None

def descriptors(train,test):
    """Four predeclared descriptors; no test target is read."""
    x=np.asarray(train['x'],float);y=np.asarray(train['y'],float);g=np.asarray(train['groups']);x0=x[:,0]
    sd=max(float(np.std(x0)),1e-8)
    td,_=support_distance(x[:,[0]],np.asarray(test['x'],float)[:,[0]])
    slopes=[];snrs=[];curves=[]
    for u in np.unique(g):
        ix=np.flatnonzero(g==u)
        if len(ix)<5: continue
        xx=(x0[ix]-np.mean(x0[ix]))/sd; yy=(y[ix]-np.mean(y[ix]))/max(float(np.std(y[ix])),1e-8)
        a=np.column_stack([np.ones(len(ix)),xx]);b=np.column_stack([np.ones(len(ix)),xx,xx**2])
        pa=a@np.linalg.lstsq(a,yy,rcond=None)[0];pb=b@np.linalg.lstsq(b,yy,rcond=None)[0]
        mse1=float(np.mean((yy-pa)**2));mse2=float(np.mean((yy-pb)**2))
        slopes.append(float(np.linalg.lstsq(a,yy,rcond=None)[0][1]))
        snrs.append(float(np.var(pb)/max(np.var(yy-pb),1e-8)))
        curves.append(float(max(0.,(mse1-mse2)/max(mse1,1e-8))))
    return {'normalized_horizon':float(np.median(td)),
            'trajectory_heterogeneity':float(median_abs_deviation(slopes,scale='normal')) if slopes else 0.,
            'degradation_snr':float(np.median(snrs)) if snrs else 0.,
            'curvature_gain':float(np.median(curves)) if curves else 0.,
            'descriptor_units':len(slopes)}

def conformal(yv,gv,pv,yt,gt,pt,dv,dt,alpha=.1):
    # Scale intervals with observed support distance; lambda is fixed at 1.
    base=max(float(np.median(np.abs(yv-pv))),1e-8);sv=base*(1+dv);st=base*(1+dt)
    score=np.abs(yv-pv)/sv;units=np.unique(gv)
    block=np.array([np.quantile(score[gv==u],1-alpha) for u in units])
    # Split-conformal finite-sample quantile over independent validation units.
    level=min(1.,np.ceil((len(block)+1)*(1-alpha))/max(len(block),1))
    q=float(np.quantile(block,level,method='higher')) if len(block) else float('nan')
    lo=np.maximum(0,pt-q*st);hi=pt+q*st;covered=(yt>=lo)&(yt<=hi)
    edges=np.quantile(dt,[0,.25,.5,.75,1]);shells=[]
    for i in range(4):
        sel=(dt>=edges[i])&((dt<=edges[i+1]) if i==3 else (dt<edges[i+1]))
        shells.append({'lo_distance':float(edges[i]),'hi_distance':float(edges[i+1]),'n':int(sel.sum()),
          'coverage':float(covered[sel].mean()) if sel.any() else None,'mean_width':float(np.mean((hi-lo)[sel])) if sel.any() else None})
    return {'nominal_coverage':1-alpha,'n_calibration_units':len(units),'block_quantile':q,
      'test_coverage':float(covered.mean()),'mean_width':float(np.mean(hi-lo)),'distance_shells':shells}

def paired(y,g,a,b,rng):
    # Positive unit delta means PP improves RMSE over plain MLP.
    units=np.unique(g);delta=np.array([np.sqrt(np.mean((y[g==u]-a[g==u])**2))-np.sqrt(np.mean((y[g==u]-b[g==u])**2)) for u in units])
    boot=np.mean(rng.choice(delta,size=(20000,len(delta)),replace=True),axis=1)
    obs=abs(delta.mean());n=len(delta)
    if n<=20:
        masks=np.arange(1<<n,dtype=np.uint64)[:,None];sign=2*(((masks>>np.arange(n,dtype=np.uint64))&1).astype(float))-1
        p=float(np.mean(np.abs(sign@delta/n)>=obs-1e-15));nperm=1<<n
    else:
        sign=rng.choice((-1.,1.),size=(200000,n));p=float((1+np.sum(np.abs(sign@delta/n)>=obs))/(len(sign)+1));nperm=len(sign)
    return {'n_units':n,'pp_unit_wins':int(np.sum(delta>0)),'mean_rmse_reduction':float(delta.mean()),
      'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'signflip_p':p,'permutations':nperm}

def fit_one(name,tr,va,te,max_epochs):
    aff=select_affine_initialization(tr,va);plain_v=[];plain_t=[];pp_v=[];pp_t=[];runs=[]
    for seed in SEEDS:
        fm=fit_plain(tr,va,seed=seed,max_epochs=max_epochs);fp=fit_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=max_epochs)
        plain_v.append(predict_plain(fm,va['x']));plain_t.append(predict_plain(fm,te['x']))
        pp_v.append(predict(fp,va['x']));pp_t.append(predict(fp,te['x']))
        runs.append({'seed':seed,'plain_test_r2':r2(te['y'],plain_t[-1]),'pp_test_r2':r2(te['y'],pp_t[-1]),
                     'plain_epoch':fm['selected_epoch'],'pp_epoch':fp.selection['selected_epoch']})
        print(name,seed,runs[-1],flush=True)
    plain_v=np.asarray(plain_v);plain_t=np.asarray(plain_t);pp_v=np.asarray(pp_v);pp_t=np.asarray(pp_t)
    dv,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);dt,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
    pe=plain_t.mean(0);ppe=pp_t.mean(0);rng=np.random.default_rng(20260908)
    result={'n':{'train':len(tr['y']),'validation':len(va['y']),'test':len(te['y']),'test_units':len(np.unique(te['groups']))},
      'descriptors':descriptors(tr,te),'runs':runs,
      'plain_ensemble':regression_metrics(te['y'],pe,te['groups']),
      'pp_ensemble':regression_metrics(te['y'],ppe,te['groups']),
      'pp_gain_r2':float(r2(te['y'],ppe)-r2(te['y'],pe)),
      'paired':paired(te['y'],te['groups'],pe,ppe,rng),
      'conformal90':conformal(va['y'],va['groups'],pp_v.mean(0),te['y'],te['groups'],ppe,dv,dt)}
    arrays={'y':te['y'],'groups':te['groups'],'plain':plain_t,'pp':pp_t,
      'validation_y':va['y'],'validation_groups':va['groups'],
      'validation_plain':plain_v,'validation_pp':pp_v,
      'validation_distance':dv,'test_distance':dt}
    return result,arrays

def fit_nasa(item,max_epochs):
    folds=item['folds'];pieces=[];arr=[]
    for fold in folds:
        r,a=fit_one('nasa_'+fold['test_cell'],fold['train'],fold['validation'],fold['test'],max_epochs);pieces.append((fold,r));arr.append(a)
    y=np.concatenate([a['y'] for a in arr]);g=np.concatenate([a['groups'] for a in arr]);plain=np.concatenate([a['plain'] for a in arr],axis=1);pp=np.concatenate([a['pp'] for a in arr],axis=1)
    vy=np.concatenate([a['validation_y'] for a in arr]);vg=np.concatenate([a['validation_groups'] for a in arr]);vp=np.concatenate([a['validation_pp'] for a in arr],axis=1);dv=np.concatenate([a['validation_distance'] for a in arr]);dt=np.concatenate([a['test_distance'] for a in arr])
    # Descriptor median across LOO folds preserves the train-only definition.
    keys=['normalized_horizon','trajectory_heterogeneity','degradation_snr','curvature_gain']
    desc={k:float(np.median([r['descriptors'][k] for _,r in pieces])) for k in keys};desc['descriptor_units']=int(np.median([r['descriptors']['descriptor_units'] for _,r in pieces]))
    pe=plain.mean(0);ppe=pp.mean(0);rng=np.random.default_rng(20260908)
    res={'n':{'train':'fold-specific','validation':len(vy),'test':len(y),'test_units':len(np.unique(g))},'descriptors':desc,
      'folds':{f['test_cell']:r for f,r in pieces},'plain_ensemble':regression_metrics(y,pe,g),'pp_ensemble':regression_metrics(y,ppe,g),
      'pp_gain_r2':float(r2(y,ppe)-r2(y,pe)),'paired':paired(y,g,pe,ppe,rng),'conformal90':conformal(vy,vg,vp.mean(0),y,g,ppe,dv,dt)}
    return res,{'y':y,'groups':g,'plain':plain,'pp':pp,
      'validation_y':vy,'validation_groups':vg,
      'validation_plain':np.concatenate([a['validation_plain'] for a in arr],axis=1),
      'validation_pp':vp,'validation_distance':dv,'test_distance':dt}

def main():
    p=argparse.ArgumentParser();p.add_argument('--max-epochs',type=int,default=300);p.add_argument('--output',default='results/cross_domain_mechanism_v1');p.add_argument('--only',default='');a=p.parse_args()
    torch.set_num_threads(2);out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);partial=out/'results.partial.json'
    payload=json.load(open(partial)) if partial.exists() else {'experiment':'cross_domain_mechanism_v1','status':'retrospective matched study','seeds':list(SEEDS),'datasets':{}}
    wanted=set(filter(None,a.only.split(',')));start=time.time()
    for name,item in all_datasets().items():
        if wanted and name not in wanted:continue
        if name in payload['datasets']:continue
        res,arr=fit_nasa(item,a.max_epochs) if isinstance(item,dict) and 'folds' in item else fit_one(name,*item,a.max_epochs)
        payload['datasets'][name]=res;np.savez_compressed(out/f'{name}_predictions.npz',**arr);partial.write_text(json.dumps(payload,indent=2)+'\n');print('DONE',name,flush=True)
    payload['runtime_seconds']=payload.get('runtime_seconds',0)+time.time()-start;(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
