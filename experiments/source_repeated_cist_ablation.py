"""Three prespecified source resplits and matched CIST route ablations."""
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.benchmark_deps'),str(ROOT/'src'),str(ROOT/'experiments')]
from source_four_model_baseline import run_model, COMMON, SEEDS
from cist_mean_alignment_source_audit import items, metric
from pp_extrapolation.model import select_affine_initialization
from pp_extrapolation.innovation_slope_transport import fit_innovation_slope_transport as fit
from pp_extrapolation.innovation_slope_transport import predict_innovation_slope_transport as predict

OUT=ROOT/'results/source_repeated_cist_ablation_v1'
SPLITS=(20260914,20260915,20260916)
MODELS=('mlp','engression','ppx_core','cist','cist_direct','cist_flow')
SETTINGS=('1P_8F','4P_1F','4P_8F','MATWI','Misata')

def run(name,setting,index,train,val,audit,seed,affine):
    if name not in ('cist_direct','cist_flow'):
        return run_model(name,setting,train,val,audit,seed,affine)
    start=time.monotonic()
    model=fit(train,val,progress_index=index,seed=seed,steps=4,beta=1.0,mean_weight=.5,
        mean_objective='zero_noise',validation_clip_to_train_range=True,validation_samples=128,
        route='direct_only' if name=='cist_direct' else 'flow_only',**COMMON)
    p=np.clip(predict(model,audit['x'],samples=128),0,max(float(train['y'].max()),1))
    assert p.shape==audit['y'].shape and np.isfinite(p).all()
    return p,dict(seed=seed,selected_epoch=model.selected_epoch,validation_mse=model.validation_mse,
        parameters=sum(v.numel() for v in model.model.parameters()),
        active_parameters=sum(v.numel() for v in model.model.parameters() if v.requires_grad),
        seconds=time.monotonic()-start,**metric(audit['y'],p,audit['groups']))

def summarize(records):
    result={}
    for model in MODELS:
        by_setting={s:[r['models'][model]['ensemble']['r2'] for r in records if r['setting']==s] for s in SETTINGS}
        values=np.array([r['models'][model]['ensemble']['r2'] for r in records])
        seed_values=np.array([x['r2'] for r in records for x in r['models'][model]['runs']])
        split_macro=[np.mean([r['models'][model]['ensemble']['r2'] for r in records if r['split']==sp]) for sp in SPLITS]
        result[model]=dict(mean_r2=float(values.mean()),minimum_split_setting_r2=float(values.min()),
            positive_split_settings=int((values>0).sum()),mean_seed_r2=float(seed_values.mean()),
            minimum_seed_r2=float(seed_values.min()),
            mean_within_split_seed_sd=float(np.mean([r['models'][model]['seed_sd'] for r in records])),
            split_macro_r2=[float(v) for v in split_macro],
            per_setting={s:dict(mean_r2=float(np.mean(v)),split_sd=float(np.std(v,ddof=1)),values=v) for s,v in by_setting.items()})
    comparisons={}
    for rival in ('engression','cist_direct','cist_flow'):
        comparisons[rival]={}
        for setting in SETTINGS:
            selected=[r for r in records if r['setting']==setting]
            delta=[r['models']['cist']['ensemble']['r2']-r['models'][rival]['ensemble']['r2'] for r in selected]
            comparisons[rival][setting]=dict(deltas=delta,wins=int(np.sum(np.array(delta)>0)),mean_delta=float(np.mean(delta)))
    return dict(models=result,cist_comparisons=comparisons)

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=False)
    files=['experiments/source_repeated_cist_ablation.py','experiments/source_four_model_baseline.py',
        'experiments/cist_mean_alignment_source_audit.py','src/pp_extrapolation/innovation_slope_transport.py']
    protocol=dict(status='source repeated-split diagnostic; external evaluation excluded',splits=SPLITS,
        seeds=SEEDS,models=MODELS,settings=SETTINGS,common=COMMON,max_fits=270,
        ablation='Only route changes; same CIST initialization, scaling, energy/MSE objective and checkpoint criterion. Inactive branches frozen.',
        limitations=['PP-X is fixed core recipes, not full executor search.',
            'Native model recipes differ in optimizer/scaling/weighting; equal epoch cap is not equal compute.',
            'Overlapping source units across resplits; resplits are not independent cohorts.',
            'One fixed hyperparameter setting; finite 60-epoch budget, not tuned final ranking.',
            'Direct-only ablation is deterministic ReLU with CIST loss/scaling, distinct from baseline tanh MLP.'],
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    for split in SPLITS:
        for setting,index,parts in items(split_seed=split):
            train,val,audit=parts
            groups=[set(p['groups']) for p in parts]
            assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
            assert all(np.var(p['y'])>0 for p in parts)
            record=dict(split=split,setting=setting,groups=[sorted(g) for g in groups],n=[len(p['y']) for p in parts],models={})
            affine=select_affine_initialization(train,val,alphas=(10.0,))
            arrays=dict(truth=audit['y'],groups=audit['groups'])
            for model in MODELS:
                ps=[];runs=[]
                for seed in SEEDS:
                    p,r=run(model,setting,index,train,val,audit,seed,affine)
                    ps.append(p);runs.append(r)
                ps=np.asarray(ps)
                arrays[model]=ps
                record['models'][model]=dict(runs=runs,ensemble=metric(audit['y'],ps.mean(0),audit['groups']),
                    seed_sd=float(np.std([r['r2'] for r in runs],ddof=1)))
                np.savez_compressed(OUT/f'{split}_{setting}.npz',**arrays)
                (OUT/'progress.json').write_text(json.dumps(dict(completed=records,current=record),indent=2))
                print('SCORED',split,setting,model,record['models'][model]['ensemble'],flush=True)
            records.append(record)
    summary=summarize(records)
    (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,records=records,summary=summary),indent=2))
    lines=['# Repeated source splits and CIST ablations','',
        'Three source resplits, three initialization seeds, max 60 epochs; no external scoring.',
        'Values below are means of the three split-specific ensemble R2 scores.',
        'Resplits reuse source machines and are not independent cohorts. PP-X denotes fixed core recipes.','',
        '| Setting | MLP | Engression | PP-X core | CIST | Direct only | Flow only |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for setting in SETTINGS:
        lines.append('| '+setting+' | '+' | '.join(f"{summary['models'][m]['per_setting'][setting]['mean_r2']:.4f}" for m in MODELS)+' |')
    lines+=['','| Model | Mean R2 | Minimum split/setting R2 | Mean seed SD |','|---|---:|---:|---:|']
    for m,v in summary['models'].items():
        lines.append(f"| {m} | {v['mean_r2']:.4f} | {v['minimum_split_setting_r2']:.4f} | {v['mean_within_split_seed_sd']:.4f} |")
    lines+=['','CIST minus comparator: positive means CIST is better.','',
        '| Setting | vs Engression | vs direct only | vs flow only |','|---|---:|---:|---:|']
    for setting in SETTINGS:
        lines.append('| '+setting+' | '+' | '.join(f"{summary['cist_comparisons'][m][setting]['mean_delta']:+.4f} ({summary['cist_comparisons'][m][setting]['wins']}/3)" for m in ('engression','cist_direct','cist_flow'))+' |')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
