#!/usr/bin/env python3
"""Retrospective v3 replay of scale-aware BQ inside the existing RBF-regime PP."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
import znion_bq_confirmatory as data
from naion_prefix_gate_eval import prefix_descriptor
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics

OUT=ROOT/"results"/"znion_scale_aware_rbf_replay_v1";SEEDS=(42,43);FINAL_SEEDS=(42,43,44,45,46)
CONFIGS=(("raw_additive",1.,"additive",.5),("norm_add_0.5",370.,"additive",.5),
         ("norm_add_2",370.,"additive",2.),("norm_add_6",370.,"additive",6.),
         ("norm_mult_0.5",370.,"multiplicative",.5),("norm_mult_1",370.,"multiplicative",1.),
         ("norm_mult_2",370.,"multiplicative",2.))

def scaled(rows,s): return {**rows,"y":np.asarray(rows["y"],np.float32)/s}

def main():
 torch.set_num_threads(2);tr,_=data.load_split("train");va,_=data.load_split("validation");dev={**tr,**va};boundary=152.;feature_scale=370.
 train=data.make_rows(tr,boundary,feature_scale,"prefix");val=data.make_rows(va,boundary,feature_scale,"tail")
 cells={p.stem:data.read_cell(p) for p in sorted((ROOT/"data/znion_rbf_regime_confirmation_v3").glob("*.xlsx"))};cells={u:c for u,c in cells.items() if c is not None and c["cycle"][-1]-c["cycle"][0]-boundary>=50};test=data.make_rows(cells,boundary,feature_scale,"tail")
 desc=np.asarray([prefix_descriptor(c["cycle"],c["capacity"],min(boundary,c["cycle"][-1]-c["cycle"][0])) for c in dev.values()]);loglife=np.log(np.asarray([c["cycle"][-1] for c in dev.values()]));center=np.median(desc,0);spread=np.maximum(np.quantile(desc,.75,axis=0)-np.quantile(desc,.25,axis=0),1e-5);memory=(desc-center)/spread
 latent=[];gates=[]
 for uid,c in sorted(cells.items()):
  q=(prefix_descriptor(c["cycle"],c["capacity"],boundary)-center)/spread;dist=((memory-q)**2).sum(1);w=np.exp(-(dist-dist.min()));life=float(np.exp(w@loglife/w.sum()));idx=np.flatnonzero(c["cycle"]-c["cycle"][0]>boundary);slope=float(np.polyfit(c["cycle"][:int(boundary)+1],c["capacity"][:int(boundary)+1],1)[0]);g=1/(1+np.exp(-slope/1e-4));latent.extend(np.maximum(life-c["cycle"][idx],0));gates.extend([g]*len(idx))
 latent=np.asarray(latent);gates=np.asarray(gates);results={}
 for name,out_scale,mode,bound in CONFIGS:
  pred=[];runs=[]
  for seed in SEEDS:
   fit=fit_boundary_quotient_pp(scaled(train,out_scale),scaled(val,out_scale),seed=seed,alpha=1000.,residual_bound=bound,correction_mode=mode,max_epochs=300,patience=50)
   base=predict_boundary_quotient(fit,scaled(test,out_scale))*out_scale;combined=(1-gates)*base+gates*latent;pred.append(combined);runs.append({"seed":seed,"epoch":fit.selection["selected_epoch"],"metrics":regression_metrics(test["y"],combined,test["groups"])})
  ensemble=np.mean(pred,axis=0);results[name]={"output_scale":out_scale,"mode":mode,"bound":bound,"runs":runs,"ensemble":regression_metrics(test["y"],ensemble,test["groups"])};print(name,results[name]["ensemble"]["pooled"]["r2"],flush=True)
 full=data.make_rows(dev,boundary,feature_scale,"prefix");pred=[];runs=[]
 for seed in FINAL_SEEDS:
  selector=fit_boundary_quotient_pp(train,val,seed=seed,alpha=1000.,residual_bound=.5,max_epochs=300,patience=50)
  epoch=max(selector.selection["selected_epoch"],1)
  fit=fit_boundary_quotient_pp(full,full,seed=seed,alpha=1000.,residual_bound=.5,max_epochs=epoch,patience=10000,restore_best=False)
  base=predict_boundary_quotient(fit,test);combined=(1-gates)*base+gates*latent;pred.append(combined);runs.append({"seed":seed,"selected_epoch":epoch,"metrics":regression_metrics(test["y"],combined,test["groups"])})
 ensemble=np.mean(pred,axis=0);results["raw_additive_full_dev_refit"]={"output_scale":1.,"mode":"additive","bound":.5,"seeds":FINAL_SEEDS,"runs":runs,"ensemble":regression_metrics(test["y"],ensemble,test["groups"])};print("raw_additive_full_dev_refit",results["raw_additive_full_dev_refit"]["ensemble"]["pooled"]["r2"],flush=True)
 result={"status":"retrospective v3 architecture replay; not confirmation","diagnostic_seeds":SEEDS,"final_refit_seeds":FINAL_SEEDS,"results":results,"selection":"none; all prespecified diagnostic arms reported","limitation":"v3 outcomes were already known and development memory includes validation EOL labels"}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(OUT/"full_dev_refit_predictions.npz",y=test["y"],groups=test["groups"],prediction=np.asarray(pred),ensemble=ensemble)

if __name__=="__main__":main()
