#!/usr/bin/env python3
"""Progress-temporal PP with validation-only opposite-ray scale transport."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from pp_extrapolation import regression_metrics
from xjtu_spectrum_adapter_v2 import load
from xjtu_uncapped_temporal_pp_v2 import rows
from xjtu_progress_temporal_pp_v3 import fit,predict
OUT=ROOT/"results/xjtu_reflected_scale_pp_v3";SEEDS=(42,43,44,45,46)

def reflected_scale(validation_truth,validation_prediction):
 denominator=float(np.dot(validation_prediction,validation_prediction))
 validation_scale=float(np.dot(validation_prediction,validation_truth)/max(denominator,1e-12))
 validation_scale=float(np.clip(validation_scale,0,2))
 # Train ray has identity scale 1. Validation and test operating conditions are
 # at coordinates -1 and +1 around train, so reflect the signed scale change.
 return validation_scale,float(np.clip(2-validation_scale,0,3))

def main():
 import torch;torch.set_num_threads(2);raw=load();tr=rows(raw,"37.5Hz11kN","summary");va=rows(raw,"35Hz12kN","summary");te=rows(raw,"40Hz10kN","summary");predictions=[];runs=[]
 for seed in SEEDS:
  fitted=fit(tr,va,seed,"pp");validation_prediction=predict(fitted,va);validation_scale,test_scale=reflected_scale(va["y"],validation_prediction);raw_prediction=predict(fitted,te);prediction=test_scale*raw_prediction;predictions.append(prediction);runs.append({"seed":seed,"validation_scale":validation_scale,"transported_test_scale":test_scale,"selection":fitted[3],"raw":regression_metrics(te["y"],raw_prediction,te["groups"]),"transported":regression_metrics(te["y"],prediction,te["groups"])});print(seed,test_scale,runs[-1]["transported"]["pooled"]["r2"],flush=True)
 predictions=np.asarray(predictions);result={"status":"post-test structural development; requires untouched replication","geometry":{"train_coordinate":0,"validation_coordinate":-1,"test_coordinate":1,"rule":"test_scale=1-(validation_scale-1)=2-validation_scale"},"scale_selection":"zero-intercept least-squares on validation only, clipped to [0,2]","runs":runs,"ensemble":regression_metrics(te["y"],predictions.mean(0),te["groups"]),"seed_mean":float(np.mean([r["transported"]["pooled"]["r2"] for r in runs])),"seed_sd":float(np.std([r["transported"]["pooled"]["r2"] for r in runs],ddof=1))}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(OUT/"predictions.npz",y=te["y"],groups=te["groups"],predictions=predictions);print(json.dumps({"ensemble":result["ensemble"]["pooled"],"seed_mean":result["seed_mean"],"seed_sd":result["seed_sd"]},indent=2))
if __name__=="__main__":main()
