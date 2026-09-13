"""Paired source-only pilot; no external evaluation or hyperparameter search."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from axial_fan_untouched_confirmation import DATA, CONFIGS, Normalizer, rows, split_units
from matwi_tool_life_untouched import DATA as MDATA, clean_tools, make_rows
from misata_machine_untouched import DATA as SDATA, prepare, rows as machine_rows
from pp_extrapolation.innovation_slope_transport import fit_innovation_slope_transport as fit
from pp_extrapolation.innovation_slope_transport import predict_innovation_slope_transport as predict

OUT = ROOT / 'results/cist_mean_alignment_source_audit_v1'
CONFIG = dict(width=32, steps=4, beta=1.0, mean_weight=0.5,
              max_epochs=60, patience=20, mean_samples=32)
SEEDS = (42, 43, 44)

def subset(data, ids):
    mask = np.isin(data['groups'], ids)
    return {k: np.asarray(v)[mask] for k, v in data.items()}

def split_ids(ids, split_seed=20260913):
    ids = np.asarray(sorted(ids))
    ids = ids[np.random.default_rng(split_seed).permutation(len(ids))]
    n = max(1, len(ids)//5)
    return ids[2*n:], ids[n:2*n], ids[:n]

def items(split_seed=20260913):
    for tag, regimes in CONFIGS.items():
        table = np.loadtxt(DATA / f'train_FAN_{tag}.txt')
        ids = np.unique(table[:, 0]).astype(int)
        a, b, c = split_ids(ids, split_seed)
        blocks = [table[np.isin(table[:, 0], part)] for part in (a,b,c)]
        norm = Normalizer.fit(blocks[0], regimes)
        parts=[]
        for block in blocks:
            part=rows(block,norm)
            mask=part['y'] <= 105
            parts.append({k:v[mask] for k,v in part.items()})
        yield tag, -1, parts
    frame=pd.read_csv(MDATA)
    identifiers=sorted(str(int(v)) for v in frame.Set.unique())
    dev=np.asarray(identifiers)[np.random.default_rng(42).permutation(len(identifiers))][:12]
    tools,_=clean_tools(frame[frame.Set.astype(int).astype(str).isin(dev)])
    ids=split_ids(list(tools), split_seed)
    maximum=float(max(len(tools[u]) for u in ids[0]))
    yield 'MATWI',9,[make_rows(tools,p,tail_fraction=.30,max_development_length=maximum) for p in ids]
    frame=pd.read_csv(SDATA)
    frame=frame[frame.split=='train'].copy()
    ids=split_ids(frame.unit_id.astype(str).unique(), split_seed)
    # Fit preprocessing only on the inner-training machines.
    marked=frame.copy()
    marked['split']=np.where(marked.unit_id.astype(str).isin(ids[0]),'train','source_holdout')
    units,modes,controls,maximum=prepare(marked)
    yield 'Misata',37,[machine_rows(units,p,modes,controls,maximum,.30,stride=3 if i==0 else 1) for i,p in enumerate(ids)]

def metric(y,p,groups):
    error=(p-y)**2
    tss=float(np.sum((y-y.mean())**2))
    return dict(r2=float(1-error.sum()/tss) if tss>0 else None,
                mse=float(error.mean()),
                group_mse=float(np.mean([error[groups==g].mean() for g in np.unique(groups)])))

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='source-only exploratory paired pilot',config=CONFIG,seeds=SEEDS,
        objectives=['zero_noise','predictive_mean'],external_scoring=False,
        note='Fixed 60-epoch budget; disjoint training, early stopping, audit machines. Not regime-held-out confirmation.',
        source_hash=hashlib.sha256((ROOT/'src/pp_extrapolation/innovation_slope_transport.py').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    for name,index,parts in items():
        train,val,audit=parts
        sets=[set(p['groups']) for p in parts]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        assert np.var(audit['y'])>0 and np.var(val['y'])>0
        record=dict(name=name,groups=[sorted(s) for s in sets],n=[len(p['y']) for p in parts],variants={})
        arrays=dict(y=audit['y'],groups=audit['groups'])
        for objective in ('zero_noise','predictive_mean'):
            predictions=[]; runs=[]
            for seed in SEEDS:
                model=fit(train,val,progress_index=index,seed=seed,mean_objective=objective,**CONFIG)
                prediction=predict(model,audit['x'],samples=32)
                assert prediction.shape==audit['y'].shape and np.isfinite(prediction).all()
                predictions.append(prediction)
                run=dict(seed=seed,epoch=model.selected_epoch,**metric(audit['y'],prediction,audit['groups']))
                runs.append(run)
                print(name,objective,run,flush=True)
            matrix=np.asarray(predictions)
            arrays[objective]=matrix
            record['variants'][objective]=dict(runs=runs,ensemble=metric(audit['y'],matrix.mean(0),audit['groups']),seed_sd=float(np.std([r['r2'] for r in runs],ddof=1)))
        results.append(record)
        np.savez_compressed(OUT/(name+'_predictions.npz'),**arrays)
        (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,settings=results),indent=2))
        print('COMPLETE',name,{k:v['ensemble'] for k,v in record['variants'].items()},flush=True)
    lines=['# CIST mean alignment: source-only pilot','', 'Fixed budget, three seeds; audit machines excluded from training and early stopping. No external scoring.','', '| Setting | Zero-noise R2 | Mean-aligned R2 | Delta |','|---|---:|---:|---:|']
    for r in results:
        a=r['variants']['zero_noise']['ensemble']['r2'];b=r['variants']['predictive_mean']['ensemble']['r2']
        lines.append(f"| {r['name']} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
