"""Post-screen numerical robustness only; frozen models, no retraining/selection."""
import torch
import numpy as np
from function_prior_train_diagnosis import ROOT
from relation_local_transport_screen import write
from transfer_error_distribution_screen import summarize


def main():
    torch.set_num_threads(2);out=ROOT/'results/component_reliability_train_v1'
    saved=dict(np.load(out/'predictions.npz'));all_results={}
    for size in (512,2048):
        u=torch.quasirandom.SobolEngine(3,scramble=True,seed=1729).draw(size).double()
        noise=torch.erfinv(2*u-1)*np.sqrt(2);episodes=[]
        for f in range(3):
            for t in (.6,.8):
                tag=f'f{f}_q{t}';p=torch.load(out/f'{tag}_prepared.pt',weights_only=False)['outer'];scores={}
                for mode in ('ridge','fixed','tied','component'):
                    ds=[]
                    for seed in ([42,43,44] if mode in ('tied','component') else [42]):
                        model=torch.load(out/f'{tag}_{mode}_{seed}.pt',weights_only=False);chunks=[]
                        with torch.no_grad():
                            for start in range(0,len(p['mean']),32):
                                batch={k:(v[start:start+32] if k in ('mean','features','margin') else v) for k,v in p.items()}
                                chunks.append(model.draws(batch,noise).numpy())
                        ds.append(np.concatenate(chunks))
                    d=np.concatenate(ds,axis=1);lo,hi=np.quantile(d,[.05,.95],axis=1)
                    scores[mode]=summarize(saved[tag+'_y'],saved[tag+'_groups'],d.mean(1),lo,hi)
                episodes.append(scores)
        all_results[str(size)]={k:{m:float(np.mean([e[k][m] for e in episodes])) for m in ('unit_mse','coverage','width','interval_score')} for k in scores}
        print(size,all_results[str(size)],flush=True)
    write(out/'quadrature_audit.json',dict(purpose='post-screen numerical robustness; not a replacement preregistered gate',
        model_parameters_frozen=True,method='scrambled Sobol normal, seed1729',results=all_results))


if __name__=='__main__':main()
