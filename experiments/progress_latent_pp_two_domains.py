#!/usr/bin/env python3
"""Scale-free progress PP for FEMTO and XJTU, selected on validation only."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from pp_extrapolation import fit_latent_pp,predict_latent
from femto_sensor_adapter_v2 import load as load_femto
from femto_corrected_benchmark_v2 import causal_features,rows,TRAIN,VAL,TEST
OUT=ROOT/"results/progress_latent_pp_two_domains_v1"; SEEDS=(42,43,44,45,46)
GRID=tuple({"width":w,"learning_rate":lr,"weight_decay":wd,"residual_bound":b,"support_decay":d} for w in (16,32) for lr in (3e-4,1e-3) for wd in (.01,.1) for b in (.5,1.) for d in (.1,.5))
def met(y,p):return {"r2":float(r2_score(y,p)),"rmse":float(mean_squared_error(y,p)**.5)}
def run(name,tr,va,te,etr,eva,ete):
    eps=1.; tt=np.log((etr+eps)/(tr["y"]+eps)); vt=np.log((eva+eps)/(va["y"]+eps))
    decoder=lambda z:np.maximum((eva+eps)*np.exp(-np.clip(z,-12,12))-eps,0)
    search=[]
    for cfg in GRID:
        f=fit_latent_pp(tr,va,tt,vt,decoder,seed=42,max_epochs=220,patience=35,**cfg)
        search.append({**cfg,"validation_rmse":f.selection["validation_rmse"]})
    chosen=min(search,key=lambda r:r["validation_rmse"]); predictions=[]; runs=[]
    for seed in SEEDS:
        f=fit_latent_pp(tr,va,tt,vt,decoder,seed=seed,max_epochs=350,patience=60,**{k:chosen[k] for k in ("width","learning_rate","weight_decay","residual_bound","support_decay")})
        latent=predict_latent(f,te["x"]); p=np.maximum((ete+eps)*np.exp(-np.clip(latent,-12,12))-eps,0); predictions.append(p);runs.append({"seed":seed,"metrics":met(te["y"],p),"selection":f.selection})
    predictions=np.asarray(predictions); result={"selected":chosen,"runs":runs,"ensemble":met(te["y"],predictions.mean(0))}
    np.savez_compressed(OUT/f"{name}.npz",y=te["y"],groups=te["groups"],predictions=predictions);print(name,result["ensemble"],flush=True);return result
def femto():
    d=load_femto();x=causal_features(d);tr,ti=rows(d,x,TRAIN);va,vi=rows(d,x,VAL);te,ei=rows(d,x,TEST,True)
    return run("femto",tr,va,te,d["elapsed_s"][ti]+10,d["elapsed_s"][vi]+10,d["elapsed_s"][ei]+10)
def xjtu():
    from xjtu_untouched import build_cache,windows
    raw=build_cache();ps=[windows(raw,n) for n in ("37.5Hz11kN","35Hz12kN","40Hz10kN")]; es=[]
    for p in ps:
        e=np.zeros(len(p["y"]));
        for u in np.unique(p["groups"]):
            ix=np.flatnonzero(p["groups"]==u);e[ix]=np.arange(8,8+len(ix))
        p["x"]=np.column_stack([p["x"],e,np.log1p(e)]).astype("float32");es.append(e)
    return run("xjtu",*ps,*es)
def main():
    import torch;torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);r={"femto":femto(),"xjtu":xjtu()};(OUT/"results.json").write_text(json.dumps(r,indent=2)+"\n")
if __name__=="__main__":main()
