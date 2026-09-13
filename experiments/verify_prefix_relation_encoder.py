import hashlib
import json
import numpy as np
import torch
from function_prior_train_diagnosis import ROOT,score
from component_transfer_screen import source_episode_indices
from relation_local_transport_screen import subset,write
from pp_extrapolation.prefix_relation_encoder import predict_prefix


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/prefix_relation_encoder_train_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    result=json.loads((out/'results.json').read_text());saved=dict(np.load(out/'predictions.npz'))
    rows=dict(np.load(ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'))
    replay=0
    for e in result['episodes']:
        fold,frac=e['fold'],e['fraction'];prefix=f'f{fold}_q{frac}'
        a,q,_,_=source_episode_indices(rows,'mich',fold,frac);query=subset(rows,q)
        assert not set(rows['groups'][a])&set(query['groups'])
        assert rows['x'][a,0].min()>query['x'][:,0].max()
        for name,seeds in [('ridge_distribution',[42]),('no_consistency',[42,43,44]),('prefix_distribution',[42,43,44])]:
            p=[]
            for seed in seeds:
                fit=torch.load(out/f'{prefix}_{name}_{seed}.pt',weights_only=False)
                d=predict_prefix(fit,query);p.append(d['mean']);replay+=1
            np.testing.assert_array_equal(np.mean(p,axis=0),saved[prefix+'_'+name])
        for name,s in e['scores'].items():
            r=score(query['y'],saved[prefix+'_'+name],query['groups'])
            for k,v in s.items():np.testing.assert_allclose(v,r[k])
            lo=saved[prefix+'_'+name+'_lo'];hi=saved[prefix+'_'+name+'_hi']
            np.testing.assert_allclose(e['interval_coverage'][name],np.mean((query['y']>=lo)&(query['y']<=hi)))
    for k in result['unit_mse']:
        np.testing.assert_allclose(result['unit_mse'][k],np.mean([e['scores'][k]['unit_mse'] for e in result['episodes']]))
    check=dict(verified=True,checkpoint_replays=replay,source_splits=6,validation_test_loaded=False)
    write(out/'verification.json',check);print(check)


if __name__=='__main__':main()
