import hashlib
import json
import numpy as np
import torch
from ravenx_train_screen import ROOT,SEEDS,ARMS,mixture_summary,ridge_summary,full_score
from relation_local_transport_screen import write
from pp_extrapolation.regime_validity_attention import predictive_summary


def main():
    torch.set_num_threads(2);out=ROOT/'results/ravenx_train_screen_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'source_audit.json').read_text())
    saved=dict(np.load(out/'predictions.npz'));replays=0
    for e,a in zip(r['episodes'],audit):
        assert not set(a['outer_source_units'])&set(a['outer_query_units']) and a['source_min']>a['query_max']
        for inner in a['inner']:
            assert not set(inner['source_units'])&set(inner['query_units'])
            assert inner['source_min']>inner['query_max'] and not inner['query_labels_fit']
        tag=f"f{e['fold']}_q{e['fraction']}";p=torch.load(out/f'{tag}_prepared.pt',weights_only=False)['outer']
        y=saved[tag+'_y'];g=saved[tag+'_groups'];base=ridge_summary(p,y)
        for field in ('mean','q05','q95','crps'):np.testing.assert_array_equal(base[field],saved[tag+'_ridge_'+field])
        for arm in ARMS:
            ds=[]
            for seed in SEEDS:
                model=torch.load(out/f'{tag}_{arm}_seed{seed}.pt',weights_only=False)
                assert sum(v.numel() for v in model.parameters())<5000
                ds.append(predictive_summary(model,p,256,1729,y,return_support=True));replays+=1
            mixed=mixture_summary(ds,y)
            for field in ('mean','q05','q95','crps'):np.testing.assert_array_equal(mixed[field],saved[tag+'_'+arm+'_'+field])
        for name,expected in e['scores'].items():
            d={k:saved[tag+'_'+name+'_'+k] for k in ('mean','q05','q95','crps')};actual=full_score(y,g,d)
            for k,v in expected.items():np.testing.assert_allclose(v,actual[k])
    for name,metrics in r['aggregate'].items():
        for metric,value in metrics.items():np.testing.assert_allclose(value,np.mean([e['scores'][name][metric] for e in r['episodes']]))
    assert not r['real_data_gate_passed'] and not r['source_gate_passed'] and not r['successor_promoted']
    report=dict(verified=True,model_replays=replays,outer_exclusion_checks=6,inner_exclusion_checks=36,
        parameter_cap_checked=True,validation_test_loaded=False,source_gate_passed=False)
    write(out/'verification.json',report);print(report)


if __name__=='__main__':main()
