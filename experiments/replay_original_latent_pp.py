"""Replay archived validation-selected PP configurations without retuning."""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np,torch
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,select_affine_initialization,regression_metrics
from regime_spline_deep_future import splits
OUT=Path('results/original_latent_pp_replay_v1')
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);old=json.load(open('results/latent_regime_deep_future_v1/results.json'));datasets=splits();summary={}
 for name in ['hust','virkler']:
  tr,va,te=datasets[name];aff=select_affine_initialization(tr,va);pred=[];runs=[]
  for archived in old['datasets'][name]['runs']:
   s=archived['selection'];seed=s['seed'];fit=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=300,patience=70,separation_weight=s['separation_weight'],gate_weight=s['gate_weight']);p,g=predict_latent_regime(fit,te['x'],return_gate=True);m=regression_metrics(te['y'],p,te['groups']);pred.append(p)
   record={'seed':seed,'selection':fit.selection,'metrics':m,'archived_r2':archived['metrics']['pooled']['r2'],'r2_delta':m['pooled']['r2']-archived['metrics']['pooled']['r2'],'epoch_matches':fit.selection['selected_epoch']==s['selected_epoch']};runs.append(record);np.savez_compressed(OUT/f'{name}_{seed}.npz',prediction=p,gate=g,y=te['y'],groups=np.asarray(te['groups'],dtype=str));torch.save({'model_state':fit.model.state_dict(),'center':fit.center,'scale':fit.scale,'target_scale':fit.target_scale,'selection':fit.selection},OUT/f'{name}_{seed}.pt');print(name,seed,m['pooled']['r2'],'delta',record['r2_delta'],flush=True)
  e=regression_metrics(te['y'],np.mean(pred,axis=0),te['groups']);rs=[r['metrics']['pooled']['r2'] for r in runs];summary[name]={'runs':runs,'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs)),'ensemble':e,'archived_ensemble_r2':old['datasets'][name]['ensemble']['pooled']['r2'],'ensemble_delta':e['pooled']['r2']-old['datasets'][name]['ensemble']['pooled']['r2']}
  (OUT/'results.json').write_text(json.dumps({'status':'reproduction of archived validation-selected configs; no retuning; no new confirmation','python':sys.version,'torch':torch.__version__,'datasets':summary},indent=2))
  print('DONE',name,e['pooled']['r2'],flush=True)
if __name__=='__main__':main()
