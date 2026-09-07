"""MATR conditional PP tuning: preserve archived loss weights, 9 trials/seed."""
import json,time
from pathlib import Path
import numpy as np,torch
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from matr_2019_latent_confirmatory import load_cells,make_rows

def main():
 torch.set_num_threads(1);out=Path('results/quick_pp_capacity_tuning_v1');out.mkdir(parents=True,exist_ok=True)
 old=json.load(open('results/matr_2019_latent_confirmatory/results.json'));audit=old['pretest'];cs={c['index']:c for c in load_cells()[0]}
 tr,va,te=[make_rows([cs[i] for i in audit['split_indices'][n]],audit['actual_boundary'],train=n=='train') for n in ['train','validation','test']];aff=select_affine_initialization(tr,va)
 grid=[dict(width=w,learning_rate=lr,weight_decay=wd) for w in [24,48,96] for lr,wd in [(2e-4,.1),(5e-4,2.),(1e-3,.01)]]
 (out/'protocol.json').write_text(json.dumps({'grid':grid,'max_epochs':300,'patience':70,'seeds':list(range(42,47)),'status':'post-hoc conditional tuning; retains previously validation-selected seed-specific PP loss weights; NOT equal total historical search budget with FT'},indent=2))
 runs=[];preds=[]
 for archived in old['runs']['latent']:
  s=archived['selection'];seed=s['seed'];dest=out/f'seed{seed}.json'
  if dest.exists():r=json.load(open(dest));p=np.load(out/f'seed{seed}.npz')['prediction']
  else:
   candidates=[];best=None
   for ci,cfg in enumerate(grid):
    start=time.monotonic();f=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=300,patience=70,separation_weight=s['separation_weight'],gate_weight=s['gate_weight'],**cfg)
    c={'config':cfg,'selection':f.selection,'seconds':time.monotonic()-start};candidates.append(c)
    if best is None or f.selection['validation_mse']<best.selection['validation_mse']:best=f;selected=cfg
    (out/f'candidates{seed}.json').write_text(json.dumps(candidates,indent=2));print('CANDIDATE',seed,ci,f.selection['validation_mse'],flush=True)
   p=predict_latent_regime(best,te['x']);r={'seed':seed,'config':selected,'selection':best.selection,'candidates':candidates,'metrics':regression_metrics(te['y'],p,te['groups'])};dest.write_text(json.dumps(r,indent=2));np.savez_compressed(out/f'seed{seed}.npz',prediction=p,y=te['y'],groups=np.asarray(te['groups'],dtype=str))
   torch.save({'model_state':best.model.state_dict(),'center':best.center,'scale':best.scale,'target_scale':best.target_scale,'selection':best.selection,'config':selected},out/f'seed{seed}.pt')
  runs.append(r);preds.append(p);print('SEED_DONE',seed,r['metrics']['pooled']['r2'],flush=True)
 rs=[r['metrics']['pooled']['r2'] for r in runs];result={'runs':runs,'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs)),'ensemble':regression_metrics(te['y'],np.mean(preds,axis=0),te['groups'])};(out/'results.json').write_text(json.dumps(result,indent=2));print('ALL_DONE',result['ensemble']['pooled']['r2'],flush=True)
if __name__=='__main__':main()
