import hashlib
import json
from dataclasses import replace
import numpy as np
import torch
from transfer_error_distribution_screen import ROOT,summarize
from component_transfer_screen import source_episode_indices
from relation_local_transport_screen import subset,write
from pp_extrapolation.prefix_relation_encoder import predict_prefix
from pp_extrapolation.transfer_error_distribution import factors


def main():
    torch.set_num_threads(2);out=ROOT/'results/transfer_error_distribution_train_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    audit=json.loads((out/'source_audit.json').read_text());r=json.loads((out/'results.json').read_text())
    saved=dict(np.load(out/'predictions.npz'));rows=dict(np.load(ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'))
    for e,audit_e in zip(r['episodes'],audit):
        f,t=e['fold'],e['fraction'];tag=f'f{f}_q{t}'
        _,q,_,_=source_episode_indices(rows,'mich',f,t);q=subset(rows,q)
        for inner in audit_e['inner']:
            s,c=set(inner['source_units']),set(inner['calibration_units']);outer=set(q['groups'])
            assert not s&c and not outer&(s|c)
            assert inner['source_min']>inner['calibration_max']
        checkpoint=torch.load(out/f'{tag}.pt',weights_only=False)
        base=predict_prefix(checkpoint['base'],q)
        np.testing.assert_array_equal(base['mean'],saved[tag+'_ridge_mean'])
        for mode in ('spread_only','global','conditional'):
            fit=checkpoint['error']
            if mode!='conditional':fit=replace(fit,residual=checkpoint['raw_error'])
            draws=base['mean'][:,None]*factors(fit,checkpoint['features'],mode)
            np.testing.assert_array_equal(draws.mean(1),saved[tag+'_'+mode+'_mean'])
            if mode=='spread_only':np.testing.assert_allclose(draws.mean(1),base['mean'])
        for mode,expected in e['scores'].items():
            actual=summarize(q['y'],q['groups'],*[saved[tag+'_'+mode+'_'+k] for k in ('mean','lo','hi')])
            for k,v in expected.items():np.testing.assert_allclose(v,actual[k])
    for mode,metrics in r['aggregate'].items():
        for metric,v in metrics.items():np.testing.assert_allclose(v,np.mean([e['scores'][mode][metric] for e in r['episodes']]))
    check=dict(verified=True,nested_exclusion_checks=36,prediction_replays=24,validation_test_loaded=False)
    write(out/'verification.json',check);print(check)


if __name__=='__main__':main()
