"""Prespecified source audit: frozen direct, point residual, integral residual."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from cist_mean_alignment_source_audit import items,metric
from pp_extrapolation.innovation_slope_transport import fit_innovation_slope_transport as fit
from pp_extrapolation.innovation_slope_transport import predict_innovation_slope_transport as predict
from pp_extrapolation.anchored_residual_transport import fit_anchored_residual,predict_anchored_residual

OUT=ROOT/'results/source_anchored_residual_audit_v1'
OLD=ROOT/'results/source_repeated_cist_ablation_v1'
SPLITS=(20260914,20260915,20260916)
SEEDS=(42,43,44)
SETTINGS=('1P_8F','4P_1F','4P_8F','MATWI','Misata')
MODELS=('direct','point_residual','integral_residual')

def summarize(records):
    out={}
    for name in MODELS:
        vals=[r['models'][name]['ensemble']['r2'] for r in records]
        out[name]=dict(mean_r2=float(np.mean(vals)),minimum_r2=float(min(vals)),
            positive_split_settings=int(np.sum(np.array(vals)>0)),
            mean_seed_sd=float(np.mean([r['models'][name]['seed_sd'] for r in records])),
            minimum_seed_r2=float(min(run['r2'] for r in records for run in r['models'][name]['runs'])),
            accepted=sum(run.get('accepted',False) for r in records for run in r['models'][name]['runs']),
            per_setting={s:dict(mean_r2=float(np.mean([r['models'][name]['ensemble']['r2'] for r in records if r['setting']==s])),
                deltas_vs_direct=[r['models'][name]['ensemble']['r2']-r['models']['direct']['ensemble']['r2'] for r in records if r['setting']==s]) for s in SETTINGS})
    return out

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=False)
    files=['experiments/source_anchored_residual_audit.py','experiments/cist_mean_alignment_source_audit.py',
        'src/pp_extrapolation/innovation_slope_transport.py','src/pp_extrapolation/anchored_residual_transport.py']
    protocol=dict(status='retrospective source-only mechanism audit',splits=SPLITS,seeds=SEEDS,
        base=dict(route='direct_only',width=32,max_epochs=60,patience=20,learning_rate=.001,weight_decay=.1,beta=1.,mean_weight=.5),
        correction=dict(width=16,steps=4,max_epochs=60,patience=20,penalty=.1,lr=.001,weight_decay=.1),
        admission=dict(min_relative_validation_mse_gain=.01,max_worst_group_mse_ratio=1.05),
        hypotheses=['Zero-start integral residual can improve frozen direct predictions.',
            'Point residual matches extra parameter count and training budget to test operator specificity.'],
        limitations=['Repeated previously inspected source splits; no untouched confirmation.',
            'Residual controls have extra capacity/training compared with direct baseline.',
            'Integral and point residuals have identical parameters but different input expansion/compute.',
            'Validation admission is not a guarantee on audit machines. No external evaluation.'],
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    for split in SPLITS:
        for setting,index,parts in items(split_seed=split):
            train,val,audit=parts
            gs=[set(p['groups']) for p in parts]
            assert all(not gs[i]&gs[j] for i in range(3) for j in range(i))
            old=np.load(OLD/f'{split}_{setting}.npz')
            assert np.array_equal(old['truth'],audit['y']) and np.array_equal(old['groups'],audit['groups'])
            rec=dict(split=split,setting=setting,groups=[sorted(g) for g in gs],models={})
            predictions={name:[] for name in MODELS};runs={name:[] for name in MODELS}
            for i,seed in enumerate(SEEDS):
                base=fit(train,val,progress_index=index,seed=seed,route='direct_only',width=32,
                    max_epochs=60,patience=20,learning_rate=.001,weight_decay=.1,steps=4,beta=1.,mean_weight=.5,
                    mean_objective='zero_noise',validation_samples=128,validation_clip_to_train_range=True)
                bp=np.clip(predict(base,audit['x'],samples=128),0,max(float(train['y'].max()),1))
                assert np.allclose(bp,old['cist_direct'][i],rtol=1e-5,atol=1e-5), 'Baseline replay mismatch'
                predictions['direct'].append(bp)
                runs['direct'].append(dict(seed=seed,selected_epoch=base.selected_epoch,**metric(audit['y'],bp,audit['groups'])))
                for route in ('point','integral'):
                    residual=fit_anchored_residual(base,train,val,seed=seed,route=route)
                    p=predict_anchored_residual(residual,audit['x'])
                    assert p.shape==bp.shape and np.isfinite(p).all()
                    if not residual.accepted:
                        assert np.allclose(p,bp,rtol=1e-5,atol=1e-5)
                    key=route+'_residual'
                    predictions[key].append(p)
                    runs[key].append(dict(seed=seed,accepted=residual.accepted,selected_epoch=residual.selected_epoch,
                        diagnostics=residual.diagnostics,**metric(audit['y'],p,audit['groups'])))
            arrays=dict(truth=audit['y'],groups=audit['groups'])
            for name in MODELS:
                ps=np.asarray(predictions[name]);arrays[name]=ps
                rec['models'][name]=dict(runs=runs[name],ensemble=metric(audit['y'],ps.mean(0),audit['groups']),
                    seed_sd=float(np.std([r['r2'] for r in runs[name]],ddof=1)))
            records.append(rec)
            np.savez_compressed(OUT/f'{split}_{setting}.npz',**arrays)
            (OUT/'progress.json').write_text(json.dumps(dict(protocol=protocol,records=records),indent=2))
            print('SCORED',split,setting,{k:dict(r2=v['ensemble']['r2'],accepted=sum(r.get('accepted',False) for r in v['runs'])) for k,v in rec['models'].items()},flush=True)
    summary=summarize(records)
    (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,records=records,summary=summary),indent=2))
    lines=['# Zero-start residual audit','',
        'Three reused source splits, three seeds. Frozen direct baseline with matched point/integral residual controls.',
        'Values are means of split-specific ensemble R2; no external confirmation.','',
        '| Setting | Direct | Point residual | Integral residual |','|---|---:|---:|---:|']
    for s in SETTINGS:
        lines.append('| '+s+' | '+' | '.join(f"{summary[m]['per_setting'][s]['mean_r2']:.4f}" for m in MODELS)+' |')
    lines+=['','| Model | Mean R2 | Worst R2 | Mean seed SD | Accepted /45 |','|---|---:|---:|---:|---:|']
    for m,v in summary.items():
        lines.append(f"| {m} | {v['mean_r2']:.4f} | {v['minimum_r2']:.4f} | {v['mean_seed_sd']:.4f} | {v['accepted']} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
