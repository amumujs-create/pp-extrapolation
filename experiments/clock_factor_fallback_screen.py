"""DS03-only structural control; semantic time factor, no novel-method claim."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pp_extrapolation.clock_factor_fallback import fit_clock,predict_clock
from pp_extrapolation.relation_local_transport import validation_acceptance
from pp_extrapolation.ds03_prospective import sha256_file
from relation_local_transport_screen import metrics,write


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/clock_factor_fallback_ds03_v1';out.mkdir(exist_ok=False)
    old=ROOT/'results/relation_local_transport_ds03_v1'
    write(out/'protocol.json',dict(primary='clock_consistent',seeds=[42,43,44],arms={'clock':0.,'clock_consistent':.1},
        status='retrospective DS03 prior-off control, not methodological novelty or strict cycle-support extrapolation',
        hypothesis='TRAIN has exact shared negative clock slope; separate unit endpoint prediction from within-unit countdown',
        selection='disjoint validation, unchanged >2%/60%/1.05 gate; no test selection',
        differences='current non-clock features feed endpoint MLP; clock enters only deterministic output; full-batch 300 epochs; extra TRAIN unit-consistency term',
        applicability='requires TRAIN-audited common affine decreasing clock; not universal PP-X replacement',
        hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'src/pp_extrapolation/clock_factor_fallback.py']}))
    train,val=[dict(np.load(old/f'{part}_rows.npz')) for part in ('train','validation')]
    baseline=dict(np.load(old/'validation_predictions.npz'))['baseline']
    fits={};vp={};decisions={}
    for arm,penalty in [('clock',0.),('clock_consistent',.1)]:
        fits[arm]=[fit_clock(train,val,seed=s,consistency=penalty) for s in (42,43,44)]
        vp[arm]=np.array([predict_clock(f,val['x']) for f in fits[arm]])
        decisions[arm]=validation_acceptance(val['y'],val['groups'],baseline.mean(0),vp[arm].mean(0))
        torch.save(fits[arm],out/f'{arm}.pt')
        print(arm,decisions[arm],flush=True)
    write(out/'selection.json',dict(decisions=decisions,logs={a:[f['selection'] for f in fs] for a,fs in fits.items()}))
    np.savez_compressed(out/'validation_predictions.npz',**vp)
    test=dict(np.load(old/'test_rows.npz'));previous=dict(np.load(old/'test_predictions.npz'))
    tp={a:np.array([predict_clock(f,test['x']) for f in fs]) for a,fs in fits.items()}
    tp.update({a:previous[a] for a in ('ppx','engression')})
    for a in fits:
        tp[a+'_guarded']=tp[a] if decisions[a]['accepted'] else tp['ppx'].copy()
    np.savez_compressed(out/'test_predictions.npz',**tp)
    scores={a:metrics(test['y'],test['groups'],p) for a,p in tp.items()}
    c=scores['clock_consistent_guarded'];gates={}
    for a in ('ppx','engression'):
        b=scores[a];gates[a]=dict(lower_rmse=c['ensemble']['rmse']<b['ensemble']['rmse'],
            no_worst_unit_loss=c['ensemble']['worst_unit_rmse']<=b['ensemble']['worst_unit_rmse'],
            no_unit_coverage_loss=c['ensemble']['positive_units']>=b['ensemble']['positive_units'],
            no_seed_coverage_loss=c['positive_seeds']>=b['positive_seeds'])
    write(out/'results.json',dict(complete=True,scores=scores,gates=gates,
        screen_passed=all(all(v.values()) for v in gates.values()),successor_promoted=False))
    print('clock control COMPLETE',flush=True)


if __name__=='__main__':
    main()
