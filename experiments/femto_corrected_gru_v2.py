#!/usr/bin/env python3
"""Corrected-channel causal sequence diagnostic for FEMTO."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from femto_sensor_adapter_v2 import load
from femto_corrected_benchmark_v2 import TRAIN,VAL,TEST
from pp_extrapolation.temporal import fit_temporal,predict_temporal
from pp_extrapolation import regression_metrics
OUT=ROOT/"results/femto_corrected_gru_v2";WINDOW=32;SEEDS=(42,43,44,45,46)
def part(d,units,endpoint=False):
    xs=[];ys=[];gs=[]
    for bearing in np.unique(d["bearing"][np.isin(d["unit"],list(units))]):
        ix=np.flatnonzero(d["bearing"]==bearing);ix=ix[np.argsort(d["recording_index"][ix])]
        base=np.column_stack([d["condition"][ix],np.log1p(d["elapsed_s"][ix])/10,d["sensor"][ix]]).astype("float32")
        use=[len(ix)-1] if endpoint else range(len(ix))
        for j in use:
            q=np.maximum(np.arange(j-WINDOW+1,j+1),0);xs.append(base[q]);ys.append(d["y"][ix[j]]);gs.append(bearing)
    n=len(xs);return {"x":np.asarray(xs),"y":np.asarray(ys,"float32"),"groups":np.asarray(gs),"prior":np.zeros((n,2),"float32"),"reliability":np.full((n,2),.5,"float32")}
def main():
    import torch;torch.set_num_threads(2);d=load();tr=part(d,TRAIN);va=part(d,VAL);te=part(d,TEST,True);runs=[];pred=[]
    for seed in SEEDS:
        f=fit_temporal(tr,va,seed=seed,mode="direct",epochs=450,patience=90);p,_=predict_temporal(f,te);pred.append(p);runs.append({"seed":seed,"selection":f["selection"],"metrics":regression_metrics(te["y"],p,te["groups"])});print(seed,runs[-1]["metrics"]["pooled"]["r2"],flush=True)
    pred=np.asarray(pred);payload={"status":"retrospective structural diagnostic","input":"corrected acceleration columns 4/5; 32-step causal sequence","runs":runs,"ensemble":regression_metrics(te["y"],pred.mean(0),te["groups"])}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(payload,indent=2)+"\n");np.savez_compressed(OUT/"predictions.npz",y=te["y"],groups=te["groups"],predictions=pred);print("ensemble",payload["ensemble"]["pooled"])
if __name__=="__main__":main()
