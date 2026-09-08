#!/usr/bin/env python3
"""Validation-selected affine shrinkage for deterministic RBF-regime PP."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
from naion_prefix_gate_eval import prefix_descriptor
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics
OUT=ROOT/'results/znion_rbf_regime_pp_alpha_development';ALPHAS=(.1,1.,10.,100.,1000.,3000.,10000.,30000.)

def eligible(directory,boundary):
 cells={p.stem:data.read_cell(p) for p in sorted(directory.glob('*.xlsx'))};return {u:c for u,c in cells.items() if c is not None and len(c['cycle'])-1-boundary>=50}

def main():
 torch.set_num_threads(2);tr,_=data.load_split('train');va,_=data.load_split('validation');dev={**tr,**va};life=[c['cycle'][-1]-c['cycle'][0] for c in tr.values()];boundary=float(np.floor(.6*np.median(life)));scale=float(max(life));train=data.make_rows(tr,boundary,scale,'prefix');validation=data.make_rows(va,boundary,scale,'tail')
 screen=[];fits={}
 for alpha in ALPHAS:
  fit=fit_boundary_quotient_pp(train,validation,seed=42,alpha=alpha,residual_bound=.5,max_epochs=300,patience=50);pred=predict_boundary_quotient(fit,validation);score=regression_metrics(validation['y'],pred,validation['groups']);screen.append({'alpha':alpha,'validation':score,'selected_epoch':fit.selection['selected_epoch']});fits[alpha]=fit
 chosen=max(screen,key=lambda x:x['validation']['pooled']['r2'])['alpha'];fit=fits[chosen]
 d=np.asarray([prefix_descriptor(c['cycle'],c['capacity'],min(boundary,c['cycle'][-1]-c['cycle'][0])) for c in dev.values()]);loglife=np.log(np.asarray([c['cycle'][-1] for c in dev.values()]));center=np.median(d,0);spread=np.maximum(np.quantile(d,.75,axis=0)-np.quantile(d,.25,axis=0),1e-5);memory=(d-center)/spread
 cohorts={'development_seen':data.load_split('test')[0],'confirmation_v2':eligible(ROOT/'data/znion_rbf_regime_confirmation_v2',boundary),'confirmation_v3':eligible(ROOT/'data/znion_rbf_regime_confirmation_v3',boundary)};results={};saved={}
 for name,cells in cohorts.items():
  rows=data.make_rows(cells,boundary,scale,'tail');base=predict_boundary_quotient(fit,rows);prediction=[];offset=0;gates={}
  for uid,c in sorted(cells.items()):
   idx=np.flatnonzero(c['cycle']-c['cycle'][0]>boundary);q=(prefix_descriptor(c['cycle'],c['capacity'],boundary)-center)/spread;distance=((memory-q)**2).sum(1);weight=np.exp(-(distance-distance.min()));life_hat=float(np.exp(np.sum(weight*loglife)/weight.sum()));slope=float(np.polyfit(c['cycle'][:int(boundary)+1],c['capacity'][:int(boundary)+1],1)[0]);gate=float(1/(1+np.exp(-slope/1e-4)));n=len(idx);prediction.extend((1-gate)*base[offset:offset+n]+gate*np.maximum(life_hat-c['cycle'][idx],0));offset+=n;gates[uid]={'gate':gate,'life_hat':life_hat}
  prediction=np.asarray(prediction);results[name]={'metrics':regression_metrics(rows['y'],prediction,rows['groups']),'gates':gates};saved[name]=(rows,prediction)
 result={'status':'post-confirmation development; alpha selected only by pre-existing development validation','selection_objective':'maximum validation pooled R2','screen':screen,'selected_alpha':chosen,'fixed':{'seed':42,'residual_bound':.5,'rbf_temperature':1.,'gate_temperature':1e-4},'replay':results,'reference_v3':{'previous_pp_pooled_r2':.558330936589257,'previous_pp_macro_r2':-.3092503976189869,'plain_mlp_pooled_r2':-.12329658655553954}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');r,p=saved['confirmation_v3'];np.savez_compressed(OUT/'v3_predictions.npz',y=r['y'],groups=r['groups'],prediction=p);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
