"""Resume the omitted setting under explicit disjoint validation units.

This is a NEW validation protocol, not completion of the historical nine-setting
table. All compared models are reselected. Test labels do not select anything.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from main9_transfer_diagnosis import (
    fit_innovation_slope_transport,predict_innovation_slope_transport,
    fit_anchored_residual,predict_anchored_residual,fit_latent_regime_pp,predict_latent_regime,
    select_affine_initialization,eng_fit,eng_predict,metric,SEEDS,
)


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/ncmapss_transfer_strict_resume_v1'
    source=ROOT/'results/main9_transfer_diagnosis_v1/ncmapss_strict_parts.npz'
    archive=np.load(source)
    tr,va,te=[{k:archive[p+'_'+k] for k in ('x','y','groups')} for p in ('train','validation','test')]
    groups=[set(p['groups']) for p in (tr,va,te)]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    assert tr['x'][:,0].max()<te['x'][:,0].min()
    assert groups[1]=={'20'}
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='new unit-disjoint validation replay, retrospective; not historical table completion',
        seeds=SEEDS,epochs=60,patience=20,groups=[sorted(g) for g in groups],
        n=[len(p['y']) for p in (tr,va,te)],input_dimensions=tr['x'].shape[1],
        limitation='one validation unit; fixed recipes, native losses and budgets; latent PP not full final multiscale PP-X',
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cap=max(float(tr['y'].max()),1.)
    affine=select_affine_initialization(tr,va)
    preds={k:[] for k in ('direct','integral','latent_pp','engression')}; runs=[]
    for seed in SEEDS:
        start=time.monotonic()
        base=fit_innovation_slope_transport(tr,va,progress_index=0,seed=seed,route='direct_only',width=32,
            max_epochs=60,patience=20,learning_rate=.001,weight_decay=.1,steps=4,beta=1.,mean_weight=.5,
            mean_objective='zero_noise',validation_samples=128,validation_clip_to_train_range=True)
        preds['direct'].append(np.clip(predict_innovation_slope_transport(base,te['x'],samples=128),0,cap))
        residual=fit_anchored_residual(base,tr,va,seed=seed,route='integral')
        preds['integral'].append(predict_anchored_residual(residual,te['x']))
        pp=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=affine,width=32,max_epochs=60,patience=20,
                              separation_weight=0.,gate_weight=0.)
        preds['latent_pp'].append(predict_latent_regime(pp,te['x']))
        eng,info=eng_fit(tr,va,seed,60,20)
        preds['engression'].append(np.clip(eng_predict(eng,te['x'],seed+20000),0,cap))
        runs.append(dict(seed=seed,direct_epoch=base.selected_epoch,residual_accepted=residual.accepted,
            residual=residual.diagnostics,latent_pp=pp.selection,engression=info,seconds=time.monotonic()-start))
        print('FIT',seed,runs[-1],flush=True)
    matrices={k:np.asarray(v) for k,v in preds.items()}
    result=dict(protocol=protocol,runs=runs,models={k:dict(ensemble=metric(te['y'],p.mean(0),te['groups']),
        seeds=[metric(te['y'],v,te['groups']) for v in p]) for k,p in matrices.items()},complete=True)
    for p in matrices.values(): assert p.shape==(3,len(te['y'])) and np.isfinite(p).all()
    (out/'results.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'predictions.npz',truth=te['y'],groups=te['groups'],**matrices)
    print('RESULT',{k:v['ensemble'] for k,v in result['models'].items()},flush=True)


if __name__=='__main__':main()
