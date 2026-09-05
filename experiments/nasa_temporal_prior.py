"""Fixed temporal prior ablation on reused NASA data, not confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_pp,predict,regression_metrics
from pp_extrapolation.temporal import fit_temporal,predict_temporal
from pp_extrapolation.hull import audit_convex_hull_support

SEEDS=(42,43,44,45,46)
ARMS=('PP_history_stats','GRU_direct','prior_only','GRU_fixed_gate','GRU_corrected')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    protocol=dict(status='post_hoc_development_reused_NASA',seeds=SEEDS,arms=ARMS,
                  contract='capacity decreasing to 1.4 Ah; same original <=1.401 label boundary',
                  short=8,long=24,history=16,hidden=16,max_epochs=300,patience=70,
                  lr=.0005,weight_decay=.05,correction_bound='.25 * max train RUL',
                  selection='validation MSE; all fixed arms reported',
                  added_information='past capacity/cycle history; old scalar PP is contextual only',
                  input_sha256=hashlib.sha256(Path(args.folds).read_bytes()).hexdigest())
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    a=np.load(args.folds,allow_pickle=False)
    folds=[{p:{k:a[f'f{i}_{p}_{k}'] for k in ('x','y','groups','prior','reliability','q','disagreement')} for p in ('train','validation','test')} for i in range(4)]
    for f in folds:
        gs=[set(f[p]['groups']) for p in ('train','validation','test')]
        assert all(gs[i].isdisjoint(gs[j]) for i,j in ((0,1),(0,2),(1,2)))
        for p in ('validation','test'):
            assert audit_convex_hull_support(f['train']['q'][:,None],f[p]['q'][:,None]).outside_fraction==1
    truth=np.concatenate([f['test']['y'] for f in folds]);groups=np.concatenate([f['test']['groups'] for f in folds])
    saved=dict(truth=truth,groups=groups,disagreement=np.concatenate([f['test']['disagreement'] for f in folds]));runs=[]
    for arm in ARMS:
        for seed in ((42,) if arm=='prior_only' else SEEDS):
            parts=[];details=[];gates=[]
            for i,f in enumerate(folds):
                cap=max(float(f['train']['y'].max()),1.)
                if arm=='prior_only':
                    g=f['test']['reliability'];p=(np.clip(f['test']['prior'],0,cap)*g).sum(1)
                    selection=dict(epoch=0)
                elif arm=='PP_history_stats':
                    tr={**f['train'],'x':f['train']['x'][:,-1,:]};va={**f['validation'],'x':f['validation']['x'][:,-1,:]}
                    fit=fit_pp(tr,va,seed=seed);p=predict(fit,f['test']['x'][:,-1,:]);g=f['test']['reliability'];selection=fit.selection
                else:
                    mode={'GRU_direct':'direct','GRU_fixed_gate':'fixed_gate','GRU_corrected':'corrected'}[arm]
                    fit=fit_temporal(f['train'],f['validation'],seed=seed,mode=mode)
                    p,g=predict_temporal(fit,f['test']);selection=fit['selection']
                parts.append(p);gates.append(g);details.append(dict(fold=i,selection=selection))
            p=np.concatenate(parts);metric=regression_metrics(truth,p,groups)
            runs.append(dict(arm=arm,seed=seed,metrics=metric,folds=details))
            saved[f'{arm}_{seed}']=p;saved[f'{arm}_{seed}_gate']=np.concatenate(gates)
            print(arm,seed,metric['pooled']['r2'],flush=True)
            (out/'progress.json').write_text(json.dumps(runs,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,runs=runs),indent=2)+'\n')
    np.savez_compressed(out/'predictions.npz',**saved)
if __name__=='__main__':main()
