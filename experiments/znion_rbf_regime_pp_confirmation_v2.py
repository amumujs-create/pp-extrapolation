#!/usr/bin/env python3
"""Deterministic RBF-regime PP and its locked second confirmation cohort."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
from naion_prefix_gate_eval import prefix_descriptor
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics

TEST_DIR=ROOT/'data/znion_rbf_regime_confirmation_v2';OUT=ROOT/'results/znion_rbf_regime_confirmation_v2'
EXPECTED=('ZN-coin_204-2_20231205230217_07_2','ZN-coin_205-3_20231205230239_07_6','ZN-coin_415-1_20231209233508_06_8','ZN-coin_428-2_20231212185058_01_4','ZN-coin_430-2_20231212185305_02_7','ZN-coin_436-2_20231227204653_03_6')
PRIOR_DIRS=('znion_bq_confirmatory/train','znion_bq_confirmatory/validation',
            'znion_bq_confirmatory/test','znion_regime_scale_confirmation_v1')

def hashes_under(paths):
 return {hashlib.sha256(p.read_bytes()).hexdigest():str(p.relative_to(ROOT)) for root in paths for p in root.glob('*.xlsx')}

def main():
 torch.set_num_threads(2);train_cells,_=data.load_split('train');val_cells,_=data.load_split('validation');dev={**train_cells,**val_cells};paths=sorted(TEST_DIR.glob('*.xlsx'))
 if tuple(p.stem for p in paths)!=EXPECTED:raise RuntimeError(f'locked files required: {EXPECTED}; found {tuple(p.stem for p in paths)}')
 prior_hash=hashes_under([ROOT/'data'/name for name in PRIOR_DIRS]);duplicates={p.stem:prior_hash[hashlib.sha256(p.read_bytes()).hexdigest()] for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() in prior_hash};seen={};internal={}
 for p in paths:
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if digest in seen:internal[p.stem]=seen[digest]
  else:seen[digest]=p.stem
 duplicates.update(internal);raw={p.stem:data.read_cell(p) for p in paths};right=[u for u,c in raw.items() if c is None];cells={u:c for u,c in raw.items() if c is not None and u not in duplicates}
 life=[c['cycle'][-1]-c['cycle'][0] for c in train_cells.values()];boundary=float(np.floor(.6*np.median(life)));scale=float(max(life));short=[u for u,c in cells.items() if c['cycle'][-1]-c['cycle'][0]-boundary<50];cells={u:c for u,c in cells.items() if u not in short}
 train=data.make_rows(train_cells,boundary,scale,'prefix');validation=data.make_rows(val_cells,boundary,scale,'tail');test=data.make_rows(cells,boundary,scale,'tail')
 descriptors=np.asarray([prefix_descriptor(c['cycle'],c['capacity'],min(boundary,c['cycle'][-1]-c['cycle'][0])) for c in dev.values()]);loglife=np.log(np.asarray([c['cycle'][-1] for c in dev.values()]));center=np.median(descriptors,0);spread=np.maximum(np.quantile(descriptors,.75,axis=0)-np.quantile(descriptors,.25,axis=0),1e-5);memory=(descriptors-center)/spread
 bq=fit_boundary_quotient_pp(train,validation,seed=42,residual_bound=.5,max_epochs=300,patience=50);base=predict_boundary_quotient(bq,test);plain_fit=fit_plain(train,validation,seed=42,max_epochs=300,patience=50);plain=predict_plain(plain_fit,test['x']);prediction=[];gates={};offset=0
 for uid,c in sorted(cells.items()):
  idx=np.flatnonzero((c['cycle']-c['cycle'][0])>boundary);q=(prefix_descriptor(c['cycle'],c['capacity'],boundary)-center)/spread;distance=((memory-q)**2).sum(1);weight=np.exp(-(distance-distance.min())/1.0);life_hat=float(np.exp(np.sum(weight*loglife)/weight.sum()));slope=float(np.polyfit(c['cycle'][:int(boundary)+1],c['capacity'][:int(boundary)+1],1)[0]);gate=float(1/(1+np.exp(-slope/1e-4)));n=len(idx);prediction.extend((1-gate)*base[offset:offset+n]+gate*np.maximum(life_hat-c['cycle'][idx],0));offset+=n;gates[uid]={'slope':slope,'gate':gate,'life_hat':life_hat}
 prediction=np.asarray(prediction);ppm=regression_metrics(test['y'],prediction,test['groups']);plm=regression_metrics(test['y'],plain,test['groups']);enough=len(cells)>=2;criteria=bool(enough and ppm['pooled']['r2']>0 and ppm['pooled']['r2']>plm['pooled']['r2']);result={'status':'one-shot untouched confirmation','frozen_model':'deterministic single-seed BQ plus RBF-attention lifetime prior','locked_candidates':EXPECTED,'eligibility':{'exact_duplicates':duplicates,'candidate_internal_duplicates':internal,'right_censored':right,'tail_shorter_than_50':short,'eligible':sorted(cells)},'n_units':len(cells),'n_rows':len(test['y']),'gates':gates,'pp':ppm,'plain_mlp':plm,'success':criteria,'criteria':'at least 2 eligible unique cells; PP pooled R2 > 0 and > single-seed plain MLP'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=test['y'],groups=test['groups'],pp=prediction,plain=plain);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
