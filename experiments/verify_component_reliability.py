import hashlib
import json
import numpy as np
import torch
from function_prior_train_diagnosis import ROOT
from relation_local_transport_screen import write
from transfer_error_distribution_screen import summarize
from pp_extrapolation.component_reliability import predict_reliability


def main():
    torch.set_num_threads(2);out=ROOT/'results/component_reliability_train_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    r=json.loads((out/'results.json').read_text());audits=json.loads((out/'audit.json').read_text())
    saved=dict(np.load(out/'predictions.npz'));replays=0
    for e,a in zip(r['episodes'],audits):
        tag=f"f{e['fold']}_q{e['fraction']}"
        for inner in a['inner']:
            s,q,o=set(inner['source']),set(inner['query']),set(a['outer_units'])
            assert not s&q and not o&(s|q) and inner['source_min']>inner['query_max']
        p=torch.load(out/f'{tag}_prepared.pt',weights_only=False)['outer']
        for mode,expected in e['scores'].items():
            draws=[]
            for seed in ([42,43,44] if mode in ('tied','component') else [42]):
                model=torch.load(out/f'{tag}_{mode}_{seed}.pt',weights_only=False)
                d,_=predict_reliability(model,p);draws.append(d);replays+=1
            d=np.concatenate(draws,axis=1)
            for key,value in [('mean',d.mean(1)),('lo',np.quantile(d,.05,axis=1)),('hi',np.quantile(d,.95,axis=1))]:
                np.testing.assert_array_equal(saved[tag+'_'+mode+'_'+key],value)
            s=summarize(saved[tag+'_y'],saved[tag+'_groups'],*[saved[tag+'_'+mode+'_'+k] for k in ('mean','lo','hi')])
            for k,v in expected.items():np.testing.assert_allclose(v,s[k])
    for mode,metrics in r['aggregate'].items():
        for k,v in metrics.items():np.testing.assert_allclose(v,np.mean([e['scores'][mode][k] for e in r['episodes']]))
    check=dict(verified=True,checkpoint_replays=replays,nested_unit_support_checks=36,external_data_loaded=False)
    write(out/'verification.json',check);print(check)


if __name__=='__main__':main()
