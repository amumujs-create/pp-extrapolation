"""Nested unit/support development screen of component reliability."""
import hashlib
import numpy as np
import torch
from function_prior_train_diagnosis import ROOT
from relation_local_transport_screen import subset,write
from component_transfer_screen import source_episode_indices
from transfer_error_distribution_screen import summarize
from pp_extrapolation.prefix_relation_encoder import fit_prefix
from pp_extrapolation.function_prior_sharing import fit_bank
from pp_extrapolation.component_reliability import prepare,fit_reliability,predict_reliability


def main():
    torch.set_num_threads(2);out=ROOT/'results/component_reliability_train_v1';out.mkdir(exist_ok=False)
    path=ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'
    write(out/'protocol.json',dict(primary='component',arms=['ridge','fixed','tied','component'],seeds=[42,43,44],
        design='six outer and six nested inner unit/health splits; gates trained only on inner queries',
        model='blend ridge coefficient Gaussian and population Gaussian componentwise; retain disagreement variance',
        gate_inputs='causal observation count, distance past donor support, relative coefficient uncertainty',
        training='120 fixed Adam steps .02; energy loss; no outer query selection',
        acceptance='component MSE >=2% better than all controls; coverage >=.9; interval score better than ridge; worst unit MSE no worse than ridge',
        limitations=['repeated TRAIN development endpoints','no coverage or identifiability guarantee','no novelty proof; no external baseline victory'],
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (path,ROOT/'experiments/component_reliability_screen.py',ROOT/'src/pp_extrapolation/component_reliability.py',
             ROOT/'src/pp_extrapolation/prefix_relation_encoder.py',ROOT/'src/pp_extrapolation/function_prior_sharing.py')}))
    rows=dict(np.load(path));results=[];audit=[];saved={}
    for f in range(3):
        for t in (.6,.8):
            a,b,_,_=source_episode_indices(rows,'mich',f,t);d,q=subset(rows,a),subset(rows,b)
            episodes=[];checks=[]
            assert not set(d['groups'])&set(q['groups']) and d['x'][:,0].min()>q['x'][:,0].max()
            for i in range(3):
                for frac in (.6,.8):
                    ia,ib,_,_=source_episode_indices(d,'mich',i,frac);src,query=subset(d,ia),subset(d,ib)
                    assert not set(src['groups'])&set(query['groups']) and src['x'][:,0].min()>query['x'][:,0].max()
                    bank=fit_bank(src);base=fit_prefix(src,bank,neural=False)
                    episodes.append(dict(prepared=prepare(base,bank,src,query),y=query['y'],groups=query['groups']))
                    checks.append(dict(source=np.unique(src['groups']).tolist(),query=np.unique(query['groups']).tolist(),
                        source_min=float(src['x'][:,0].min()),query_max=float(query['x'][:,0].max())))
            bank=fit_bank(d);base=fit_prefix(d,bank,neural=False);p=prepare(base,bank,d,q)
            tag=f'f{f}_q{t}';torch.save(dict(episodes=episodes,outer=p),out/f'{tag}_prepared.pt')
            scores={};gates={}
            saved[tag+'_y']=q['y'];saved[tag+'_groups']=q['groups']
            for mode in ('ridge','fixed','tied','component'):
                draws=[];weights=[]
                for seed in ([42,43,44] if mode in ('tied','component') else [42]):
                    model=fit_reliability(episodes,mode,seed)
                    v,w=predict_reliability(model,p);draws.append(v);weights.append(w)
                    torch.save(model,out/f'{tag}_{mode}_{seed}.pt')
                v=np.concatenate(draws,axis=1);pred=v.mean(1);lo,hi=np.quantile(v,[.05,.95],axis=1)
                scores[mode]=summarize(q['y'],q['groups'],pred,lo,hi)
                gates[mode]=np.mean(weights,axis=(0,1)).tolist()
                for k,value in [('mean',pred),('lo',lo),('hi',hi)]:saved[tag+'_'+mode+'_'+k]=value
            results.append(dict(fold=f,fraction=t,scores=scores,mean_gate=gates))
            audit.append(dict(outer_units=np.unique(q['groups']).tolist(),inner=checks))
            print(tag,{k:round(v['unit_mse'],4) for k,v in scores.items()},flush=True)
    aggregate={k:{m:float(np.mean([e['scores'][k][m] for e in results])) for m in ('unit_mse','coverage','width','interval_score')} for k in scores}
    worst={k:max(max(e['scores'][k]['unit_mse_values']) for e in results) for k in scores}
    c=aggregate['component'];r=aggregate['ridge']
    accepted=all(c['unit_mse']<.98*aggregate[k]['unit_mse'] for k in scores if k!='component') and c['coverage']>=.9 and c['interval_score']<r['interval_score'] and worst['component']<=worst['ridge']
    np.savez_compressed(out/'predictions.npz',**saved);write(out/'audit.json',audit)
    write(out/'results.json',dict(complete=True,episodes=results,aggregate=aggregate,worst_unit_mse=worst,source_gate_passed=accepted,successor_promoted=False))
    print('COMPLETE',aggregate,'accepted',accepted,flush=True)


if __name__=='__main__':main()
