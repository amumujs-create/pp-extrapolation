"""Check hashes, TRAIN-only support splits, prediction replay and all saved scores."""
import hashlib
import json
import numpy as np
import torch
from function_prior_train_diagnosis import ROOT,score,donor_predictions,own_prediction
from pp_extrapolation.function_prior_sharing import fit_bank
from component_transfer_screen import source_episode_indices
from relation_local_transport_screen import subset,write


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/function_prior_train_diagnosis_v3'
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    train=dict(np.load(ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'))
    result=json.loads((out/'results.json').read_text());saved=dict(np.load(out/'predictions.npz'))
    count=0
    for e in result['episodes']:
        fold,frac=e['fold'],e['fraction'];prefix=f'f{fold}_q{frac}'
        a,q,cut,_=source_episode_indices(train,'mich',fold,frac)
        donors,query=subset(train,a),subset(train,q)
        early=subset(train,train['x'][:,0]>=-cut)
        assert not set(donors['groups'])&set(query['groups'])
        assert early['x'][:,0].min()>query['x'][:,0].max()
        assert len(early['y'])==e['early_rows']<len(train['y'])
        np.testing.assert_array_equal(saved[prefix+'_y'],query['y'])
        np.testing.assert_array_equal(saved[prefix+'_groups'],query['groups'])
        matrix=donor_predictions(fit_bank(donors),query)
        np.testing.assert_allclose(matrix,saved[prefix+'_donor_predictions'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(own_prediction(fit_bank(early),query,allow_missing=True),saved[prefix+'_own_early_label_oracle'],rtol=1e-10,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(matrix.mean(1),saved[prefix+'_uniform_function_mean'])
        oracle=saved[prefix+'_oracle_best_single_donor']
        for u in np.unique(query['groups']):
            ix=query['groups']==u;best=((matrix[ix]-query['y'][ix,None])**2).mean(0).argmin()
            np.testing.assert_allclose(matrix[ix,best],oracle[ix])
        for k,value in e['scores'].items():
            recomputed=score(query['y'],saved[prefix+'_'+k],query['groups'])
            for field in value:np.testing.assert_allclose(value[field],recomputed[field],rtol=1e-12,atol=1e-12)
            count+=1
    for k,v in result['aggregate_unit_mse'].items():
        np.testing.assert_allclose(v,np.mean([e['scores'][k]['unit_mse'] for e in result['episodes']]))
    assert not result['successor_promoted']
    checks=dict(verified=True,episodes=6,score_replays=count,bank_refits=12,validation_test_loaded=False,
        oracle_labels_separated=True,old_v1_invalid=True)
    write(out/'verification.json',checks);print(checks)


if __name__=='__main__':main()
