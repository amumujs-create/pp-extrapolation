#!/usr/bin/env python3
"""Development-only evaluation of distance--uncertainty PP shrinkage."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

from pp_extrapolation import (
    fit_pp, predict, regression_metrics, select_affine_initialization,
    support_distance, component_ensemble, combine_uncertainty_gated,
    select_uncertainty_gate, predict_uncertainty_gated,
)
from pp_extrapolation.model import _affine_prediction

SEEDS=(42,43,44,45,46)
BETAS=(0.,.05,.1,.25,.5,1.,2.,4.,8.)
GAMMAS=(0.,1.,2.,4.,8.,16.,32.,64.)

def metrics(y,p,g): return regression_metrics(y,np.asarray(p),g)
def seed_summary(y, matrix, groups):
    scores=[metrics(y,p,groups) for p in matrix]
    values=np.asarray([x['pooled']['r2'] for x in scores])
    return {'pooled_r2_mean':float(values.mean()),'pooled_r2_sd':float(values.std()),
            'ensemble':metrics(y,np.mean(matrix,axis=0),groups),'per_seed':scores}

def evaluate_split(name,train,val,test,epochs):
    vd,va=support_distance(train['x'][:,[0]],val['x'][:,[0]])
    td,ta=support_distance(train['x'][:,[0]],test['x'][:,[0]])
    if va.outside_fraction<1 or ta.outside_fraction<1: raise RuntimeError(f'{name}: non-strict hull split')
    affine=select_affine_initialization(train,val)
    fits=[fit_pp(train,val,seed=s,affine_selection=affine,max_epochs=epochs) for s in SEEDS]
    selection=select_uncertainty_gate(fits,val['x'],val['y'],vd,betas=BETAS,gammas=GAMMAS)
    distance_only=select_uncertainty_gate(fits,val['x'],val['y'],vd,betas=BETAS,gammas=(0.,))
    _,test_residual,test_u,cap=component_ensemble(fits,test['x'])
    test_affine,_,_,_=component_ensemble(fits,test['x'])
    pp=np.asarray([predict(f,test['x']) for f in fits])
    sg=combine_uncertainty_gated(test_affine,test_residual,td,test_u,beta=distance_only.beta,gamma=0,output_cap=cap)
    du=predict_uncertainty_gated(fits,test['x'],td,selection)
    ridge=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale'])
    print(name,'beta/gamma',selection.beta,selection.gamma,'PP',metrics(test['y'],pp.mean(0),test['groups'])['pooled']['r2'],
          'DU',metrics(test['y'],du.mean(0),test['groups'])['pooled']['r2'],flush=True)
    return {'n':{k:len(v['y']) for k,v in [('train',train),('validation',val),('test',test)]},
      'hull':{'validation':va.summary(),'test':ta.summary()},
      'selection':{'beta':selection.beta,'gamma':selection.gamma,'validation_mse':selection.validation_mse,
                   'validation_ensemble_mse':selection.validation_ensemble_mse,
                   'distance_only_beta':distance_only.beta,'candidates':list(selection.candidates)},
      'test_uncertainty':{'mean':float(test_u.mean()),'p95':float(np.quantile(test_u,.95))},
      'ridge':metrics(test['y'],ridge,test['groups']),'pp':seed_summary(test['y'],pp,test['groups']),
      'distance_pp':seed_summary(test['y'],sg,test['groups']),'du_pp':seed_summary(test['y'],du,test['groups'])}

def battery_features(part,scale):
    z=np.asarray(part['x'],dtype=np.float32); h=z[:,:,0]; r=z[:,:,1]
    return {'x':np.column_stack((h[:,-1],r[:,-1],h.mean(1),r.mean(1),h[:,-1]-h[:,0],r[:,-1]-r[:,0],part['cycles']/scale)).astype(np.float32),
      'y':np.asarray(part['y'],dtype=np.float32),'groups':np.asarray(part['units'])}

def prepare_battery(raw):
    endpoint=raw['train']['x'][:,-1,0]; cutoff=float(np.quantile(endpoint,.25)); tr=endpoint>cutoff; va=raw['val']['x'][:,-1,0]<cutoff
    scale=max(float(raw['train']['cycles'].max()),1.)
    subset=lambda p,m: {k:v[m] for k,v in p.items()}
    return battery_features(subset(raw['train'],tr),scale),battery_features(subset(raw['val'],va),scale),battery_features(raw['source'],scale)

def evaluate_nasa(folds,epochs):
    collected={k:[] for k in ('truth','groups','ridge')}; matrices={k:[[] for _ in SEEDS] for k in ('pp','distance_pp','du_pp')}; choices=[]
    for fold in folds:
        train,val,test=fold['train'],fold['validation'],fold['test']; vd,va=support_distance(train['x'][:,[0]],val['x'][:,[0]]); td,ta=support_distance(train['x'][:,[0]],test['x'][:,[0]])
        affine=select_affine_initialization(train,val); fits=[fit_pp(train,val,seed=s,affine_selection=affine,max_epochs=epochs) for s in SEEDS]
        du_sel=select_uncertainty_gate(fits,val['x'],val['y'],vd,betas=BETAS,gammas=GAMMAS); d_sel=select_uncertainty_gate(fits,val['x'],val['y'],vd,betas=BETAS,gammas=(0.,))
        aa,rr,uu,cap=component_ensemble(fits,test['x']); pp=np.asarray([predict(f,test['x']) for f in fits])
        sg=combine_uncertainty_gated(aa,rr,td,uu,beta=d_sel.beta,gamma=0,output_cap=cap); du=predict_uncertainty_gated(fits,test['x'],td,du_sel)
        collected['truth'].append(test['y']); collected['groups'].append(test['groups']); collected['ridge'].append(_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale']))
        for i in range(len(SEEDS)):
            for key,arr in [('pp',pp),('distance_pp',sg),('du_pp',du)]: matrices[key][i].append(arr[i])
        choices.append({'test_cell':fold['test_cell'],'beta':du_sel.beta,'gamma':du_sel.gamma,'distance_only_beta':d_sel.beta,
                        'validation_mse':du_sel.validation_mse,'validation_ensemble_mse':du_sel.validation_ensemble_mse,
                        'hull':{'validation':va.summary(),'test':ta.summary()}})
    y=np.concatenate(collected['truth']); g=np.concatenate(collected['groups']); ridge=np.concatenate(collected['ridge'])
    joined={k:np.asarray([np.concatenate(parts) for parts in per_seed]) for k,per_seed in matrices.items()}
    print('nasa choices',[(x['beta'],x['gamma']) for x in choices],flush=True)
    return {'selection_by_fold':choices,'ridge':metrics(y,ridge,g),**{k:seed_summary(y,v,g) for k,v in joined.items()}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--legacy-root',default=str(Path(__file__).resolve().parents[2]/'ca-css-ncmapss'))
    p.add_argument('--max-epochs',type=int,default=300); p.add_argument('--output',default='results/distance_uncertainty_pp_v1'); a=p.parse_args()
    sys.path.insert(0,str(Path(a.legacy_root).resolve()))
    from run_affine_tail_external_three import prepare_hust,prepare_virkler
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from pae_boundary_realdata import prepare_dataset
    torch.set_num_threads(2); start=time.time(); datasets={}
    for name,prepared in [('hust',prepare_hust()),('virkler',prepare_virkler())]:
        datasets[name]=evaluate_split(name,*prepared[:3],a.max_epochs)
    for name in ('rwth','mich'):
        raw,_=prepare_dataset(name); datasets[name]=evaluate_split(name,*prepare_battery(raw),a.max_epochs)
    folds,_=prepare_folds(); datasets['nasa']=evaluate_nasa(folds,a.max_epochs)
    payload={'experiment':'distance_uncertainty_pp_v1','status':'development_only','selection_data':'validation labels only; XJTU and MATR excluded',
      'formula':'affine + exp(-beta*distance-gamma*seed_disagreement)*NN_residual','seeds':list(SEEDS),'datasets':datasets,'runtime_seconds':time.time()-start}
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
