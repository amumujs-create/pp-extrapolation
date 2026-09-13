"""Post-v1 sticky remediation on byte-identical prepared TRAIN-only episodes."""
import hashlib
import json
import numpy as np
import torch
from ravenx_train_screen import ROOT,SEEDS,ARMS,mixture_summary,full_score,score
from relation_local_transport_screen import write
from pp_extrapolation.regime_validity_attention import predictive_summary
from pp_extrapolation.sticky_regime_validity_attention import fit_sticky_raven


def main():
    torch.set_num_threads(2);old=ROOT/'results/ravenx_train_screen_v1';out=ROOT/'results/ravenx_sticky_train_screen_v2';out.mkdir(exist_ok=False)
    files=(ROOT/'src/pp_extrapolation/sticky_regime_validity_attention.py',ROOT/'protocols/RAVENX_STICKY_REMEDIATION_PROTOCOL.md')
    prepared={}
    for f in range(3):
        for q in (.6,.8):
            p=old/f'f{f}_q{q}_prepared.pt';prepared[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(out/'protocol.json',dict(primary='ravenx',version=2,seeds=SEEDS,arms=ARMS,steps=80,
        reason='v1 mean no-switch probability 0.00962 exposed missing sticky duration prior',
        evaluation='256 Sobol per seed, exact equal-seed path mixture',validation_test_loaded=False,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},prepared_hashes=prepared))
    old_result=json.loads((old/'results.json').read_text());old_saved=dict(np.load(old/'predictions.npz'))
    episodes=[];saved={}
    for old_episode in old_result['episodes']:
        f,q=old_episode['fold'],old_episode['fraction'];tag=f'f{f}_q{q}'
        item=torch.load(old/f'{tag}_prepared.pt',weights_only=False);inner,p=item['inner'],item['outer']
        y=old_saved[tag+'_y'];g=old_saved[tag+'_groups'];scores={'ridge':old_episode['scores']['ridge']};diagnostics={};logs={}
        for field in ('mean','q05','q95','crps'):saved[tag+'_ridge_'+field]=old_saved[tag+'_ridge_'+field]
        for arm,configuration in ARMS.items():
            ds=[];diagnostics[arm]=[];logs[arm]=[]
            for seed in SEEDS:
                model,log=fit_sticky_raven(inner,configuration['config'],seed,80,configuration['counterfactual'])
                d=predictive_summary(model,p,256,1729,y,return_support=True);ds.append(d);logs[arm].append(log)
                diagnostics[arm].append(dict(no_switch=float(d['path_probability'][:,0].mean()),
                    attention=np.mean(d['attention'],axis=0).tolist(),edge_gate=d['edge_gate'].tolist()))
                torch.save(model,out/f'{tag}_{arm}_seed{seed}.pt')
            mixed=mixture_summary(ds,y);scores[arm]=full_score(y,g,mixed)
            for field in ('mean','q05','q95','crps'):saved[tag+'_'+arm+'_'+field]=mixed[field]
        saved[tag+'_y']=y;saved[tag+'_groups']=g
        episodes.append(dict(fold=f,fraction=q,scores=scores,diagnostics=diagnostics,training=logs))
        write(out/'partial.json',dict(episodes=episodes));print('SCORE',tag,{k:round(v['unit_mse'],4) for k,v in scores.items()},flush=True)
    names=list(episodes[0]['scores']);metrics=('unit_mse','coverage','width','interval_score','crps','positive_units')
    aggregate={k:{m:float(np.mean([e['scores'][k][m] for e in episodes])) for m in metrics} for k in names}
    worst={k:max(max(e['scores'][k]['unit_mse_values']) for e in episodes) for k in names}
    seed_wins=0
    for seed in SEEDS:
        candidate=[];baseline=[]
        for e in episodes:
            tag=f"f{e['fold']}_q{e['fraction']}";p=torch.load(old/f'{tag}_prepared.pt',weights_only=False)['outer']
            model=torch.load(out/f'{tag}_ravenx_seed{seed}.pt',weights_only=False)
            candidate.append(score(saved[tag+'_y'],predictive_summary(model,p,256,1729)['mean'],saved[tag+'_groups'])['unit_mse'])
            baseline.append(e['scores']['ridge']['unit_mse'])
        seed_wins+=int(np.mean(candidate)<.98*np.mean(baseline))
    r=aggregate['ravenx'];controls=[k for k in names if k!='ravenx']
    real=(r['unit_mse']<.98*aggregate['ridge']['unit_mse'] and all(r['unit_mse']<.95*aggregate[k]['unit_mse'] for k in controls if k!='ridge') and
        all(r['crps']<aggregate[k]['crps'] for k in controls) and worst['ravenx']<=worst['ridge'] and .85<=r['coverage']<=.95 and
        r['interval_score']<aggregate['ridge']['interval_score'] and r['positive_units']>=aggregate['ridge']['positive_units'] and seed_wins>=4)
    np.savez_compressed(out/'predictions.npz',**saved)
    write(out/'results.json',dict(complete=True,aggregate=aggregate,worst_unit_mse=worst,episodes=episodes,seed_wins_over_ridge=seed_wins,
        real_data_gate_passed=real,synthetic_gate_passed=True,source_gate_passed=real,successor_promoted=False,
        decision='advance to validation' if real else 'stop v2 before validation'))
    print('COMPLETE',json.dumps(aggregate,indent=2),'gate',real,flush=True)


if __name__=='__main__':main()
