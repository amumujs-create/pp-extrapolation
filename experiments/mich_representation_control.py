"""Same-row MICH 11-feature control; excludes pooled training and full refit."""
from pathlib import Path
import json
import hashlib
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src'),str(ROOT.parent/'ca-css-ncmapss')]
from main9_transfer_diagnosis import (SEEDS,BUDGETS,metric,eng_fit,eng_predict,
    fit_innovation_slope_transport,predict_innovation_slope_transport,
    fit_anchored_residual,predict_anchored_residual,fit_latent_regime_pp,predict_latent_regime,
    select_affine_initialization)
from pae_boundary_realdata import prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,FEATURE_NAMES
from distance_uncertainty_pp import prepare_battery


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/mich_representation_control_v1';out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='post-test same-row feature control; no full PP-X comparison',
        seeds=SEEDS,budgets=BUDGETS,features=FEATURE_NAMES,
        limitation='MICH-only training, original filtered rows, no train+validation refit; historical BQ used three datasets',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    raw,audit=prepare_dataset('mich');old=prepare_battery(raw)
    cutoff=float(np.quantile(raw['train']['x'][:,-1,0],.25))
    sub=lambda p,m:{k:v[m] for k,v in p.items()}
    parts=[sub(raw['train'],raw['train']['x'][:,-1,0]>cutoff),
           sub(raw['val'],raw['val']['x'][:,-1,0]<cutoff),raw['source']]
    scale=BatteryRepresentationScale.fit(parts[0],audit['boundary'])
    rows=[]
    for previous,part in zip(old,parts):
        assert np.array_equal(previous['y'],part['y']) and np.array_equal(previous['groups'],part['units'])
        rows.append(dict(x=scale.features(part),y=previous['y'],groups=previous['groups']))
    tr,va,te=rows;cap=max(float(tr['y'].max()),1.)
    assert te['x'][:,0].max()<tr['x'][:,0].min()
    gs=[set(p['groups']) for p in rows];assert all(not gs[i]&gs[j] for i in range(3) for j in range(i))
    affine=select_affine_initialization(tr,va)
    records=[]
    for epochs in BUDGETS:
        matrices={n:[] for n in ('direct','integral','latent_pp','engression')}; runs=[]
        for seed in SEEDS:
            start=time.monotonic();patience=20 if epochs==60 else 80
            base=fit_innovation_slope_transport(tr,va,progress_index=0,seed=seed,route='direct_only',
                width=32,max_epochs=epochs,patience=patience,learning_rate=.001,weight_decay=.1,
                steps=4,beta=1.,mean_weight=.5,mean_objective='zero_noise',validation_samples=128,
                validation_clip_to_train_range=True)
            matrices['direct'].append(np.clip(predict_innovation_slope_transport(base,te['x'],samples=128),0,cap))
            residual=fit_anchored_residual(base,tr,va,seed=seed,route='integral')
            matrices['integral'].append(predict_anchored_residual(residual,te['x']))
            pp=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=affine,width=32,
                max_epochs=epochs,patience=patience,separation_weight=0.,gate_weight=0.)
            matrices['latent_pp'].append(predict_latent_regime(pp,te['x']))
            eng,info=eng_fit(tr,va,seed,epochs,patience)
            matrices['engression'].append(np.clip(eng_predict(eng,te['x'],seed+20000),0,cap))
            runs.append(dict(seed=seed,direct_epoch=base.selected_epoch,integral_accepted=residual.accepted,
                integral=residual.diagnostics,latent_pp=pp.selection,engression=info,seconds=time.monotonic()-start))
        matrices={k:np.asarray(v) for k,v in matrices.items()}
        record=dict(epochs=epochs,runs=runs,models={k:dict(ensemble=metric(te['y'],p.mean(0),te['groups']),
            seeds=[metric(te['y'],v,te['groups']) for v in p]) for k,p in matrices.items()})
        for p in matrices.values():assert p.shape==(3,len(te['y'])) and np.isfinite(p).all()
        records.append(record)
        (out/'results.json').write_text(json.dumps(dict(protocol=protocol,records=records,complete=len(records)==2),indent=2))
        np.savez_compressed(out/f'predictions_{epochs}.npz',truth=te['y'],groups=te['groups'],**matrices)
        print('RESULT',epochs,{k:v['ensemble']['r2'] for k,v in record['models'].items()},flush=True)


if __name__=='__main__':main()
