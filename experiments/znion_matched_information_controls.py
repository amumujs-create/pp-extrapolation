#!/usr/bin/env python3
"""Matched-information controls for the developed Zn-ion regime PP."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
from naion_prefix_gate_eval import prefix_descriptor
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import regression_metrics
OUT=ROOT/'results/znion_matched_information_controls';SEEDS=(42,43,44,45,46)

def main():
 torch.set_num_threads(2);tr,_=data.load_split('train');va,_=data.load_split('validation');dev={**tr,**va};boundary=152.;scale=370.
 cells={p.stem:data.read_cell(p) for p in sorted((ROOT/'data/znion_rbf_regime_confirmation_v3').glob('*.xlsx'))};cells={u:c for u,c in cells.items() if c is not None and len(c['cycle'])-1-boundary>=50};test=data.make_rows(cells,boundary,scale,'tail')
 train=data.make_rows(tr,boundary,scale,'prefix');validation=data.make_rows(va,boundary,scale,'tail');original=[];unbounded=[];cap_hits=[]
 for seed in SEEDS:
  fit=fit_plain(train,validation,seed=seed,max_epochs=300,patience=50);capped=predict_plain(fit,test['x']);raw=predict_plain(fit,test['x'],clip_to_train_max=False);original.append(capped);unbounded.append(raw);cap_hits.append(int(np.sum(raw>fit['target_scale'])))
 full=data.make_rows(dev,boundary,scale,'prefix');train_tail=data.make_rows(tr,boundary,scale,'tail');refit=[];epochs=[]
 for seed in SEEDS:
  fit=fit_plain(full,train_tail,seed=seed,max_epochs=500,patience=70);refit.append(predict_plain(fit,test['x'],clip_to_train_max=False));epochs.append(fit['selected_epoch'])
 desc=np.asarray([prefix_descriptor(c['cycle'],c['capacity'],min(boundary,len(c['cycle'])-1)) for c in dev.values()]);loglife=np.log(np.asarray([c['cycle'][-1] for c in dev.values()]));center=np.median(desc,0);spread=np.maximum(np.quantile(desc,.75,axis=0)-np.quantile(desc,.25,axis=0),1e-5);memory=(desc-center)/spread;lifetime=[]
 for uid,c in sorted(cells.items()):
  q=(prefix_descriptor(c['cycle'],c['capacity'],boundary)-center)/spread;distance=((memory-q)**2).sum(1);weight=np.exp(-(distance-distance.min()));life_hat=float(np.exp(weight@loglife/weight.sum()));idx=np.flatnonzero(c['cycle']-c['cycle'][0]>boundary);lifetime.extend(np.maximum(life_hat-c['cycle'][idx],0))
 pp_file=np.load(ROOT/'results/znion_rbf_regime_pp_alpha_development/v3_predictions.npz',allow_pickle=True);assert np.array_equal(pp_file['groups'],test['groups']);arms={'original_mlp_seed_mean':np.mean(original,0),'original_mlp_unbounded_seed_mean':np.mean(unbounded,0),'matched_dev_refit_mlp_seed_mean':np.mean(refit,0),'lifetime_rbf_only':np.asarray(lifetime),'developed_regime_pp':pp_file['prediction']}
 result={'status':'post-hoc matched-information audit on already observed v3','protocol':{'task':'online RUL after age boundary','boundary':boundary,'test_units':sorted(cells),'mlp_refit':'all development prefixes; original-train tails for epoch selection','warning':'refit validation shares physical units with its training prefix and is a fairness control, not an untouched model selection protocol'},'original_train_target_max':float(train['y'].max()),'original_mlp_upper_clip_activations_by_seed':cap_hits,'refit_selected_epochs':epochs,'metrics':{name:regression_metrics(test['y'],pred,test['groups']) for name,pred in arms.items()}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=test['y'],groups=test['groups'],**arms);print(json.dumps({'cap_hits':cap_hits,'scores':{k:(v['pooled']['r2'],v['unit_macro_r2']) for k,v in result['metrics'].items()}},indent=2))
if __name__=='__main__':main()
