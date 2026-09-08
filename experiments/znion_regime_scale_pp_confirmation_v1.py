#!/usr/bin/env python3
"""One-shot evaluation of the frozen regime-scale PP on locked Zn-ion cells."""
from __future__ import annotations
import json,sys
from pathlib import Path
import hashlib
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
from znion_regime_scale_pp_development import LifetimeScaleHead
from naion_prefix_gate_eval import prefix_descriptor
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics

TEST_DIR=ROOT/'data/znion_regime_scale_confirmation_v1';OUT=ROOT/'results/znion_regime_scale_confirmation_v1';SEEDS=(42,43,44,45,46)
EXPECTED=('ZN-coin_410-2_20231209232626_09_2','ZN-coin_412-2_20231209233028_09_8','ZN-coin_439-1_20231227204804_04_6')

def main():
 torch.set_num_threads(2);train_cells,_=data.load_split('train');val_cells,_=data.load_split('validation');dev={**train_cells,**val_cells};paths=sorted(TEST_DIR.glob('*.xlsx'))
 if tuple(p.stem for p in paths)!=EXPECTED:raise RuntimeError(f'locked test files required: {EXPECTED}; found {tuple(p.stem for p in paths)}')
 test_cells={p.stem:data.read_cell(p) for p in paths};right_censored=[u for u,c in test_cells.items() if c is None];test_cells={u:c for u,c in test_cells.items() if c is not None}
 life=[c['cycle'][-1]-c['cycle'][0] for c in train_cells.values()];boundary=float(np.floor(.6*np.median(life)));scale=float(max(life));no_tail=[u for u,c in test_cells.items() if c['cycle'][-1]-c['cycle'][0]<=boundary]
 train=data.make_rows(train_cells,boundary,scale,'prefix');validation=data.make_rows(val_cells,boundary,scale,'tail');test=data.make_rows(test_cells,boundary,scale,'tail')
 descriptors=np.asarray([prefix_descriptor(c['cycle'],c['capacity'],min(boundary,c['cycle'][-1]-c['cycle'][0])) for c in dev.values()],np.float32);target=np.log(np.asarray([c['cycle'][-1] for c in dev.values()],np.float32));center=descriptors.mean(0);spread=descriptors.std(0);spread[spread<1e-6]=1;x=torch.tensor((descriptors-center)/spread);y=torch.tensor(target)
 pp,plain,runs=[],[],[]
 for seed in SEEDS:
  bq=fit_boundary_quotient_pp(train,validation,seed=seed,residual_bound=.5,max_epochs=300,patience=50);bq_pred=predict_boundary_quotient(bq,test);plain_fit=fit_plain(train,validation,seed=seed,max_epochs=300,patience=50);plain_pred=predict_plain(plain_fit,test['x']);plain.append(plain_pred)
  torch.manual_seed(seed);head=LifetimeScaleHead();opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
  for _ in range(2000):loss=(head(x)-y).square().mean();opt.zero_grad();loss.backward();opt.step()
  mixed=[];offset=0
  for uid,c in sorted(test_cells.items()):
   if uid in no_tail:continue
   idx=np.flatnonzero((c['cycle']-c['cycle'][0])>boundary);d=prefix_descriptor(c['cycle'],c['capacity'],boundary);life_hat=float(np.exp(head(torch.tensor(((d-center)/spread)[None],dtype=torch.float32)).item()));latent=np.maximum(life_hat-c['cycle'][idx],0);slope=float(np.polyfit(c['cycle'][:int(boundary)+1],c['capacity'][:int(boundary)+1],1)[0]);gate=float(1/(1+np.exp(-slope/1e-4)));n=len(idx);mixed.extend((1-gate)*bq_pred[offset:offset+n]+gate*latent);offset+=n
  pp.append(mixed);runs.append({'seed':seed,'pp':regression_metrics(test['y'],np.asarray(mixed),test['groups']),'plain_mlp':regression_metrics(test['y'],plain_pred,test['groups'])})
 pp=np.asarray(pp);plain=np.asarray(plain);ppm=regression_metrics(test['y'],pp.mean(0),test['groups']);plm=regression_metrics(test['y'],plain.mean(0),test['groups']);prior=list((ROOT/'data/znion_bq_confirmatory/test').glob('*.xlsx'));prior_hashes={hashlib.sha256(p.read_bytes()).hexdigest():p.name for p in prior};duplicates={p.stem:prior_hashes[hashlib.sha256(p.read_bytes()).hexdigest()] for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() in prior_hashes};criteria_met=bool(ppm['pooled']['r2']>0 and ppm['pooled']['r2']>plm['pooled']['r2']);result={'status':'one-shot outcome; source-duplicate audit prevents independent-confirmation claim' if duplicates else 'one-shot untouched confirmation','frozen_from_commit':'7edc6ae','locked_files':EXPECTED,'right_censored':right_censored,'no_evaluable_tail':no_tail,'exact_duplicates_of_prior_test':duplicates,'n_evaluation_units':len(np.unique(test['groups'])),'n_rows':len(test['y']),'runs':runs,'pp_mean':ppm,'pp_median':regression_metrics(test['y'],np.median(pp,axis=0),test['groups']),'plain_mean':plm,'success_criteria_met':criteria_met,'independent_confirmation':bool(criteria_met and not duplicates),'criteria':'PP mean pooled R2 > 0 and > plain MLP mean pooled R2'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=test['y'],groups=test['groups'],pp=pp,plain=plain);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
