"""Paired original PP affine ablation; archived hyperparameters, no retuning."""
import json
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from matr_2019_latent_confirmatory import load_cells,make_rows

def main():
 torch.set_num_threads(1)
 out=Path('results/quick_original_affine_ablation_v1');out.mkdir(parents=True,exist_ok=True)
 old=json.load(open('results/matr_2019_latent_confirmatory/results.json'));audit=old['pretest'];cs={c['index']:c for c in load_cells()[0]}
 tr,va,te=[make_rows([cs[i] for i in audit['split_indices'][n]],audit['actual_boundary'],train=n=='train') for n in ['train','validation','test']]
 aff=select_affine_initialization(tr,va);results={}
 for adaptive in [False,True]:
  name='trainable_affine' if adaptive else 'original_frozen';runs=[];preds=[]
  for archived in old['runs']['latent']:
   s=archived['selection'];seed=s['seed'];dest=out/f'{name}_{seed}.json'
   if dest.exists():
    r=json.load(open(dest));p=np.load(out/f'{name}_{seed}.npz')['prediction']
   else:
    f=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=300,patience=70,separation_weight=s['separation_weight'],gate_weight=s['gate_weight'],trainable_affine=adaptive)
    p=predict_latent_regime(f,te['x']);r={'seed':seed,'selection':f.selection,'metrics':regression_metrics(te['y'],p,te['groups']),'archived_validation_mse':s['validation_mse']}
    np.savez_compressed(out/f'{name}_{seed}.npz',prediction=p,y=te['y'],groups=np.asarray(te['groups'],dtype=str));dest.write_text(json.dumps(r,indent=2))
   runs.append(r);preds.append(p);print(name,seed,r['metrics']['pooled']['r2'],flush=True)
  rs=[r['metrics']['pooled']['r2'] for r in runs];results[name]={'runs':runs,'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs)),'ensemble':regression_metrics(te['y'],np.mean(preds,axis=0),te['groups'])}
  (out/'results.json').write_text(json.dumps(results,indent=2));print('DONE',name,results[name]['ensemble']['pooled']['r2'],flush=True)
if __name__=='__main__':main()
