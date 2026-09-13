"""Fixed MICH diagnostic: data pool x boundary parameterization x refit.

No new architecture or successor promotion. MICH test rows are identical in
every arm. The joint dual-scale refit is checked against archived PP-X seeds.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.benchmark_deps'),str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part
from pp_extrapolation.presets import battery_dual_scale_pp_config
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
from engression.engression import Engressor

SEEDS=(42,43,44)
POOLS=('mich_filtered','mich_all','joint_all')
MODELS=('direct_softplus','bq_fixed','bq_dual','engression')
MAX_EPOCHS=500
PATIENCE=70


def subset(rows,mask):return {k:v[mask] for k,v in rows.items()}


def no_boundary(rows):
    # x still contains the same declared margin. Only output factorization and
    # its supervised affine-target initialization change from y/m to y.
    return {**rows,'margin':np.ones_like(rows['margin'])}


def cfg(name):
    if name=='bq_dual':return battery_dual_scale_pp_config()
    return dict(width=64,alpha=1000.,learning_rate=.001,weight_decay=.01,residual_bound=2.)


def eng_prediction(model,rows,seed):
    # Fixed row chunks bound memory; the same rule is used for every arm.
    preds=[]
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(seed)
        for start in range(0,len(rows['y']),512):
            p=model.predict_onebatch(torch.tensor(rows['x'][start:start+512],dtype=torch.float32),target='mean',sample_size=128)
            preds.append(p.cpu().numpy().reshape(-1))
    return np.maximum(np.concatenate(preds),0.)


def fit_eng(train,val,seed,epochs,select=True):
    torch.manual_seed(seed)
    model=Engressor(train['x'].shape[1],1,num_layer=2,hidden_dim=64,noise_dim=32,
                    beta=1.,lr=.001,standardize=True,device='cpu',check_device=False,verbose=False)
    x=torch.tensor(train['x'],dtype=torch.float32);y=torch.tensor(train['y'][:,None],dtype=torch.float32)
    best=float('inf');best_epoch=0;state=None
    if not select:
        for _ in range(epochs):
            model.train(x,y,num_epoches=1,batch_size=min(512,len(x)),verbose=False)
        return model,dict(selected_epoch=epochs,selection=False)
    for epoch in range(1,epochs+1):
        model.train(x,y,num_epoches=1,batch_size=min(512,len(x)),verbose=False)
        pred=eng_prediction(model,val,10000+seed)
        err=(pred-val['y'])**2
        score=float(np.mean([err[val['dataset']==d].mean() for d in np.unique(val['dataset'])]))
        if score<best-1e-8:best,best_epoch,state=score,epoch,copy.deepcopy(model.model.state_dict())
        if epoch-best_epoch>PATIENCE:break
    model.model.load_state_dict(state)
    return model,dict(selected_epoch=best_epoch,validation_dataset_macro_mse=best,executed_epochs=epoch)


def score(rows,matrix,scales):
    result={}
    for di in np.unique(rows['dataset']):
        name=DATASETS[int(di)];mask=rows['dataset']==di
        y=rows['y'][mask].astype(float)*scales[name].time_scale
        p=matrix[:,mask].astype(float)*scales[name].time_scale
        g=rows['units'][mask]
        def metric(v):
            err=(v-y)**2
            per=[float(np.sqrt(err[g==u].mean())) for u in np.unique(g)]
            return dict(r2=float(1-err.sum()/np.sum((y-y.mean())**2)),rmse=float(np.sqrt(err.mean())),
                        unit_rmse=float(np.mean(per)),worst_unit_rmse=max(per))
        result[name]=dict(ensemble=metric(p.mean(0)),seeds=[metric(v) for v in p],
            positive_seed_r2=sum(metric(v)['r2']>0 for v in p),
            positive_units=int(sum(1-np.sum((p.mean(0)[g==u]-y[g==u])**2)/np.sum((y[g==u]-y[g==u].mean())**2)>0 for u in np.unique(g))))
    return result


def group_ids(rows):return set(zip(rows['dataset'].tolist(),rows['units'].tolist()))


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/mich_pool_boundary_refit_factorial_v1'
    resume='--resume' in sys.argv
    out.mkdir(parents=True,exist_ok=resume)
    paths=('experiments/mich_pool_boundary_refit_factorial.py','src/pp_extrapolation/boundary_quotient.py',
           'src/pp_extrapolation/presets.py','.benchmark_deps/engression/engression.py')
    protocol=dict(status='retrospective fixed factorial diagnosis; no new-model novelty or promotion',
        seeds=SEEDS,pools=POOLS,models=MODELS,refit=(False,True),max_epochs=MAX_EPOCHS,patience=PATIENCE,
        features=11,selection='equal-dataset normalized validation MSE; no test-based selection',
        refit_rule='reinitialize from seed and train full pool for max(selected_epoch,1), no checkpoint selection',
        common='same rows/features/target normalization within arm; all outputs nonnegative, no upper clipping',
        native_differences='BQ AdamW and equal-dataset/unit loss; official Engression Adam and native row loss; same epoch cap is not equal compute',
        boundary_control='identical frozen affine + bounded SiLU network; unit margin disables factor and quotient-target initialization together',
        scales='fixed per-dataset original train scales, shared across all arms; filtered arm still uses these source-train-only constants',
        filtered='original MICH health cutoff train/validation rows; refit uses their union only',
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    if resume:
        protocol=json.loads((out/'protocol.json').read_text())
        (out/'resume.json').write_text(json.dumps(dict(
            reason='serialize coverage count as Python int; before first Engression run align refit one-epoch train calls with selection',
            current_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    else:
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    scales={}; prepared={}; raw_splits={}
    for i,name in enumerate(DATASETS):
        s,a=prepare_dataset(name);raw_splits[name]=s
        scales[name]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
        prepared[name]={p:build_rows(rows,scales[name],i) for p,rows in (
            ('train',s['train']),('validation',s['val']),('full',full_part(s)),('test',s['source']))}
    joint={p:concatenate_rows([prepared[n][p] for n in DATASETS]) for p in ('train','validation','full','test')}
    mich=prepared['mich']; raw=raw_splits['mich'];cut=float(np.quantile(raw['train']['x'][:,-1,0],.25))
    filtered=dict(train=subset(mich['train'],raw['train']['x'][:,-1,0]>cut),
                  validation=subset(mich['validation'],raw['val']['x'][:,-1,0]<cut),test=mich['test'])
    filtered['full']=concatenate_rows([filtered['train'],filtered['validation']])
    pools=dict(mich_filtered=filtered,mich_all=mich,joint_all=joint)
    archived=np.load(ROOT/'results/bq_dual_scale_final_replay_v1/predictions.npz')
    assert np.array_equal(joint['test']['y'],archived['y'])
    assert np.array_equal(joint['test']['units'],archived['units'])
    assert np.array_equal(joint['test']['dataset'],archived['dataset'])
    audit={}
    for pool,parts in pools.items():
        tr,va,te=[parts[p] for p in ('train','validation','test')]
        gs=[group_ids(p) for p in (tr,va,te)]
        assert all(not gs[i]&gs[j] for i in range(3) for j in range(i))
        assert np.array_equal(subset(te,te['dataset']==DATASETS.index('mich'))['y'],mich['test']['y'])
        audit[pool]=dict(n={p:len(r['y']) for p,r in parts.items()},units={p:len(group_ids(r)) for p,r in parts.items()})
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    records=json.loads((out/'results.json').read_text())['records'] if resume and (out/'results.json').exists() else []
    for pool,parts in pools.items():
        for name in MODELS:
            if sum(r['pool']==pool and r['model']==name for r in records)==2:continue
            matrices={False:[],True:[]};runs=[]
            for seed in SEEDS:
                start=time.monotonic()
                tr,va,full,te=[parts[p] for p in ('train','validation','full','test')]
                if name=='engression':
                    model,selection=fit_eng(tr,va,seed,MAX_EPOCHS)
                    before=eng_prediction(model,te,20000+seed)
                    ep=max(selection['selected_epoch'],1)
                    final,_=fit_eng(full,full,seed,ep,select=False)
                    after=eng_prediction(final,te,20000+seed)
                else:
                    if name=='direct_softplus':tr,va,full,te=map(no_boundary,(tr,va,full,te))
                    model=fit_boundary_quotient_pp(tr,va,seed=seed,max_epochs=MAX_EPOCHS,patience=PATIENCE,**cfg(name))
                    selection=model.selection
                    before=predict_boundary_quotient(model,te)
                    ep=max(selection['selected_epoch'],1)
                    final=fit_boundary_quotient_pp(full,full,seed=seed,max_epochs=ep,patience=10000,restore_best=False,**cfg(name))
                    after=predict_boundary_quotient(final,te)
                for p in (before,after):assert p.shape==te['y'].shape and np.isfinite(p).all()
                matrices[False].append(before);matrices[True].append(after)
                runs.append(dict(seed=seed,selection=selection,refit_epochs=ep,seconds=time.monotonic()-start))
                print('FIT',pool,name,seed,'epoch',ep,'sec',round(time.monotonic()-start,1),flush=True)
            for refit,values in matrices.items():
                mat=np.asarray(values)
                record=dict(pool=pool,model=name,refit=refit,runs=runs,metrics=score(parts['test'],mat,scales))
                if pool=='joint_all' and name=='bq_dual' and refit:
                    delta=float(np.max(np.abs(mat-archived['prediction'][:len(SEEDS)])))
                    record['archived_replay_max_abs_normalized_error']=delta
                    record['archived_replay_matches']=bool(np.allclose(mat,archived['prediction'][:len(SEEDS)],rtol=1e-5,atol=1e-5))
                records.append(record)
                saved_path=out/f'{pool}_{name}_refit{int(refit)}.npz'
                if saved_path.exists():
                    previous=np.load(saved_path)
                    np.testing.assert_allclose(previous['prediction'],mat,rtol=1e-6,atol=1e-6)
                else:
                    np.savez_compressed(saved_path,prediction=mat,
                                        y=parts['test']['y'],units=parts['test']['units'],dataset=parts['test']['dataset'])
                print('RESULT',pool,name,refit,record['metrics']['mich']['ensemble'],flush=True)
            (out/'results.json').write_text(json.dumps(dict(protocol=protocol,records=records,complete=len(records)==24),indent=2))
    (out/'scales.json').write_text(json.dumps({n:s.metadata() for n,s in scales.items()},indent=2))


if __name__=='__main__':main()
