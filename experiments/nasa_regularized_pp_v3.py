"""Validation-selected target-perturbation regularization for NASA PP."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from nasa_causal_multiscale_pp import folds
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
OUT=ROOT/'results/nasa_regularized_pp_v3';SEEDS=range(42,47);LEVELS=(0.,.05,.1,.2)

def perturbed(rows,level,seed):
 q={k:(v.copy() if hasattr(v,'copy') else v) for k,v in rows.items()};rng=np.random.default_rng(seed+int(level*10000))
 q['y']=np.clip(q['y']+rng.normal(0,level*np.std(q['y']),len(q['y'])),0,None).astype(np.float32);return q

def train(f,level,seed,sep):
 tr=perturbed(f['train'],level,seed);a=select_affine_initialization(tr,f['validation'])
 return fit_latent_regime_pp(tr,f['validation'],seed=seed,affine_selection=a,max_epochs=400,patience=80,separation_weight=sep,gate_weight=0.,width=32)

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);arch=json.load(open(ROOT/'results/nasa_causal_multiscale_pp_v1/results.json'))
 ys=[];gs=[];pred=[[] for _ in SEEDS];audit=[]
 for fi,old in enumerate(arch['folds']):
  preset=old['selected']['preset'];sep=old['selected']['separation'];f=folds(preset)[fi];screen=[]
  for level in LEVELS:
   fit=train(f,level,42,sep);screen.append({'level':level,'validation_mse':fit.selection['validation_mse'],'epoch':fit.selection['selected_epoch']})
  selected=min(screen,key=lambda q:q['validation_mse']);runs=[]
  for j,seed in enumerate(SEEDS):
   fit=train(f,selected['level'],seed,sep);p=predict_latent_regime(fit,f['test']['x']);pred[j].append(p);runs.append({'seed':seed,**fit.selection})
  ys.append(f['test']['y']);gs.append(f['test']['groups']);audit.append({'test_cell':f['test_cell'],'preset':preset,'separation':sep,'screen':screen,'selected':selected,'runs':runs});print(fi,selected,flush=True)
 y=np.concatenate(ys);g=np.concatenate(gs);P=np.asarray([np.concatenate(q) for q in pred]);result={'status':'retrospective development; perturbation selected per fold using validation only',
  'levels':LEVELS,'folds':audit,'ensemble':regression_metrics(y,P.mean(0),g),'per_seed':[regression_metrics(y,p,g) for p in P],
  'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=P,y=y,groups=g);print('FINAL',result['ensemble']['pooled'],flush=True)
if __name__=='__main__':main()
