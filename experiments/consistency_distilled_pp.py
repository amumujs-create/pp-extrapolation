#!/usr/bin/env python3
"""Distil a multi-seed PP consensus into one deployable PP."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from plain_mlp_ablation import fit_plain,predict_plain
from distance_uncertainty_pp import prepare_battery

SEEDS=(42,43,44,45,46); LAMBDAS=(.01,.1,1.)
def score(y,p,g): return regression_metrics(y,p,g)
def summary(y,mat,g):
    r=np.asarray([score(y,p,g)['pooled']['r2'] for p in mat])
    return {'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std()),'ensemble':score(y,mat.mean(0),g),'per_seed_r2':r.tolist()}

def run_split(name,tr,va,te,epochs):
    affine=select_affine_initialization(tr,va)
    teachers=[fit_pp(tr,va,seed=s,affine_selection=affine,max_epochs=epochs) for s in SEEDS]
    # Distillation sees training coordinates only; validation remains label/feature held out.
    teacher_ref=np.mean([predict(f,tr['x']) for f in teachers],axis=0)
    candidates=[]; students_by_lambda={}
    for lam in LAMBDAS:
        students=[fit_pp(tr,va,seed=s,affine_selection=affine,max_epochs=epochs,
          consistency_x=tr['x'],consistency_target=teacher_ref,consistency_weight=lam) for s in SEEDS]
        mse=float(np.mean([(predict(f,va['x'])-va['y'])**2 for f in students]))
        candidates.append({'lambda':lam,'mean_seed_validation_mse':mse}); students_by_lambda[lam]=students
    best=min(candidates,key=lambda x:(x['mean_seed_validation_mse'],x['lambda'])); students=students_by_lambda[best['lambda']]
    pp=np.asarray([predict(f,te['x']) for f in teachers]); distilled=np.asarray([predict(f,te['x']) for f in students])
    plain=np.asarray([predict_plain(fit_plain(tr,va,seed=s,max_epochs=epochs),te['x']) for s in SEEDS])
    print(name,'lambda',best['lambda'],'PP',summary(te['y'],pp,te['groups'])['pooled_r2_mean'],'CD',summary(te['y'],distilled,te['groups'])['pooled_r2_mean'],flush=True)
    return {'selected_lambda':best['lambda'],'lambda_candidates':candidates,'pp':summary(te['y'],pp,te['groups']),
            'consistency_pp':summary(te['y'],distilled,te['groups']),'plain_nn':summary(te['y'],plain,te['groups']),
            '_arrays':{'y':te['y'],'g':te['groups'],'pp':pp,'consistency_pp':distilled,'plain_nn':plain}}

def run_nasa(folds,epochs):
    stores={k:[[] for _ in SEEDS] for k in ('pp','consistency_pp','plain_nn')}; ys=[]; gs=[]; selections=[]
    for fold in folds:
        d=run_split('NASA-'+fold['test_cell'],fold['train'],fold['validation'],fold['test'],epochs)
        a=d.pop('_arrays');ys.append(a['y']);gs.append(a['g'])
        for k in stores:
            for i in range(len(SEEDS)): stores[k][i].append(a[k][i])
        selections.append(d)
    y=np.concatenate(ys);g=np.concatenate(gs)
    return {'fold_results':selections,**{k:summary(y,np.asarray([np.concatenate(v) for v in parts]),g) for k,parts in stores.items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',default=str(Path(__file__).resolve().parents[2]/'ca-css-ncmapss'));p.add_argument('--max-epochs',type=int,default=300);p.add_argument('--output',default='results/consistency_distilled_pp_v1');a=p.parse_args()
    sys.path.insert(0,str(Path(a.legacy_root).resolve()))
    from run_affine_tail_external_three import prepare_hust,prepare_virkler
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from pae_boundary_realdata import prepare_dataset
    torch.set_num_threads(2);start=time.time();data={}
    for n,pd in [('hust',prepare_hust()),('virkler',prepare_virkler())]: data[n]=run_split(n,*pd[:3],a.max_epochs);data[n].pop('_arrays')
    for n in ('rwth','mich'):
        raw,_=prepare_dataset(n);data[n]=run_split(n,*prepare_battery(raw),a.max_epochs);data[n].pop('_arrays')
    folds,_=prepare_folds();data['nasa']=run_nasa(folds,a.max_epochs)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps({'experiment':'consistency_distilled_pp_v1','deployment_models':1,'teacher_models_during_training':5,'datasets':data,'runtime_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
