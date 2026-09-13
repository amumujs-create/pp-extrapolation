"""Fixed-budget source audit of native model recipes, with disjoint machines."""
from pathlib import Path
import sys
import copy
import hashlib
import json
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.benchmark_deps'), str(ROOT/'src'), str(ROOT/'experiments')]
from cist_mean_alignment_source_audit import items, metric
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation.model import fit_pp, predict, select_affine_initialization
from pp_extrapolation.innovation_slope_transport import fit_innovation_slope_transport, predict_innovation_slope_transport
from engression.engression import Engressor

OUT = ROOT/'results/source_four_model_baseline_v1'
SEEDS = (42,43,44)
MODELS = ('mlp','engression','ppx_core','cist')
COMMON = dict(width=32,learning_rate=1e-3,weight_decay=.1,max_epochs=60,patience=20)

def eng_prediction(model, x, seed):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        p=model.predict_onebatch(torch.as_tensor(x,dtype=torch.float32),target='mean',sample_size=128)
    return p.cpu().numpy().reshape(-1)

def fit_eng(train, validation, seed):
    torch.manual_seed(seed)
    model=Engressor(train['x'].shape[1],1,num_layer=2,hidden_dim=32,noise_dim=32,
        beta=1.0,lr=1e-3,standardize=True,device='cpu',check_device=False,verbose=False)
    x=torch.as_tensor(train['x'],dtype=torch.float32)
    y=torch.as_tensor(train['y'][:,None],dtype=torch.float32)
    cap=max(float(train['y'].max()),1.0)
    best=float('inf');epoch_best=0;state=None
    for epoch in range(1,61):
        # The installed implementation and Adam state are retained across epochs.
        model.train(x,y,num_epoches=1,batch_size=min(512,len(x)),verbose=False)
        p=np.clip(eng_prediction(model,validation['x'],100000+seed),0,cap)
        mse=float(np.mean((p-validation['y'])**2))
        if mse < best-1e-10:
            best=mse;epoch_best=epoch;state=copy.deepcopy(model.model.state_dict())
        if epoch-epoch_best>20:
            break
    model.model.load_state_dict(state)
    return model, epoch_best, best, epoch

def run_model(name, setting, train, validation, audit, seed, affine):
    start=time.monotonic()
    cap=max(float(train['y'].max()),1.0)
    if name=='mlp':
        fit=fit_plain(train,validation,seed=seed,**COMMON)
        p=predict_plain(fit,audit['x'])
        ep=fit['selected_epoch'];loss=fit['validation_mse'];net=fit['model']
    elif name=='ppx_core':
        gated=setting not in ('MATWI','Misata')
        fit=fit_pp(train,validation,seed=seed,affine_selection=affine,
            learned_affine_gate=gated,direct_residual_mixture=gated,**COMMON)
        p=predict(fit,audit['x'])
        ep=fit.selection['selected_epoch'];loss=fit.selection['best_validation_mse'];net=fit.model
    elif name=='engression':
        fit,ep,loss,_=fit_eng(train,validation,seed)
        p=eng_prediction(fit,audit['x'],200000+seed);net=fit.model
    else:
        fit=fit_innovation_slope_transport(train,validation,seed=seed,
            progress_index=-1 if setting not in ('MATWI','Misata') else (9 if setting=='MATWI' else 37),
            steps=4,beta=1.0,mean_weight=.5,mean_objective='zero_noise',
            validation_clip_to_train_range=True,validation_samples=128,**COMMON)
        p=predict_innovation_slope_transport(fit,audit['x'],samples=128)
        ep=fit.selected_epoch;loss=fit.validation_mse;net=fit.model
    p=np.clip(p,0,cap)
    assert p.shape==audit['y'].shape and np.isfinite(p).all()
    scores=metric(audit['y'],p,audit['groups'])
    return p,dict(seed=seed,selected_epoch=int(ep),validation_mse=float(loss),
        parameters=sum(v.numel() for v in net.parameters()),seconds=time.monotonic()-start,**scores)

def summary(settings):
    result={}
    for name in MODELS:
        rs=[s['models'][name] for s in settings]
        values=np.array([r['ensemble']['r2'] for r in rs])
        result[name]=dict(mean_setting_r2=float(values.mean()),
            cohort_equal_r2=float(np.mean([values[:3].mean(),values[3],values[4]])),
            minimum_r2=float(values.min()),positive_settings=int((values>0).sum()),
            mean_seed_r2=float(np.mean([r['mean_seed_r2'] for r in rs])),
            minimum_seed_r2=float(min(r['minimum_seed_r2'] for r in rs)),
            mean_seed_sd=float(np.mean([r['seed_sd'] for r in rs])))
    return result

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=False)
    paths=['experiments/source_four_model_baseline.py','experiments/cist_mean_alignment_source_audit.py',
        'experiments/plain_mlp_ablation.py','src/pp_extrapolation/model.py',
        'src/pp_extrapolation/innovation_slope_transport.py','.benchmark_deps/engression/engression.py']
    protocol=dict(status='fixed-budget source audit; previously inspected source split',seeds=SEEDS,
        common=COMMON,inference_samples=128,external_scoring=False,hyperparameter_search=False,
        clipping='[0, max(train RUL, 1)] for validation and audit',
        ppx_core='Axial learned affine gate/direct mixture; MATWI and Misata unbounded PP core; ridge alpha=10 fixed. Full PP-X executor search excluded.',
        differences=['Native feature/target scaling and activation retained.',
            'MLP/PP/CIST use equal-unit weights and AdamW; Engression uses native row weights and Adam.',
            'Same epoch cap is not equal FLOPs or parameter count; no claim of fully tuned ranking.',
            'Engression native train called for one epoch at a time with persistent Adam; its end-of-call energy evaluation consumes RNG.',
            'CIST retains zero-noise MSE; this is a baseline comparison, not a new architecture.'],
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    settings=[]
    for setting,_,parts in items():
        train,val,audit=parts
        groups=[set(p['groups']) for p in parts]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
        assert np.var(val['y'])>0 and np.var(audit['y'])>0
        record=dict(name=setting,groups=[sorted(g) for g in groups],
            n=[len(p['y']) for p in parts],validation_y_range=[float(val['y'].min()),float(val['y'].max())],models={})
        affine=select_affine_initialization(train,val,alphas=(10.0,))
        arrays=dict(truth=audit['y'],groups=audit['groups'])
        for name in MODELS:
            ps=[];runs=[]
            for seed in SEEDS:
                p,run=run_model(name,setting,train,val,audit,seed,affine)
                ps.append(p);runs.append(run)
                print('RUN',setting,name,run,flush=True)
            ps=np.asarray(ps);rs=np.asarray([r['r2'] for r in runs])
            arrays[name]=ps
            record['models'][name]=dict(runs=runs,ensemble=metric(audit['y'],ps.mean(0),audit['groups']),
                mean_seed_r2=float(rs.mean()),minimum_seed_r2=float(rs.min()),seed_sd=float(rs.std(ddof=1)))
            np.savez_compressed(OUT/(setting+'_predictions.npz'),**arrays)
            (OUT/'progress.json').write_text(json.dumps(dict(completed=settings,current=record),indent=2))
            print('SCORED',setting,name,record['models'][name]['ensemble'],flush=True)
        settings.append(record)
    payload=dict(protocol=protocol,settings=settings,summary=summary(settings))
    (OUT/'results.json').write_text(json.dumps(payload,indent=2))
    lines=['# Source four-model baseline','',
        'Three independent initialization seeds, 60-epoch cap. Disjoint fit/early-stop/audit machines. Source development evidence only.',
        'PP-X core denotes the fixed recipes in protocol.json; this is not a full tuned executor comparison.',
        'Native optimizers, target scaling and unit weighting differ; shared epoch cap does not imply equal compute.','',
        '| Setting | MLP | Engression | PP-X core | CIST |','|---|---:|---:|---:|---:|']
    for s in settings:
        lines.append('| '+s['name']+' | '+' | '.join(f"{s['models'][m]['ensemble']['r2']:.4f}" for m in MODELS)+' |')
    lines+=['','| Model | Macro R2 | Cohort-equal R2 | Minimum R2 | Seed SD |','|---|---:|---:|---:|---:|']
    for m,s in payload['summary'].items():
        lines.append(f"| {m} | {s['mean_setting_r2']:.4f} | {s['cohort_equal_r2']:.4f} | {s['minimum_r2']:.4f} | {s['mean_seed_sd']:.4f} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('SUMMARY',json.dumps(payload['summary']),flush=True)

if __name__=='__main__':
    main()
