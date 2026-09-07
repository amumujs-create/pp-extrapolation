#!/usr/bin/env python3
from pathlib import Path
import json,sys,time
import numpy as np,torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'.benchmark_deps'),str(R/'src'),str(R/'experiments')]
from engression import engression
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,select_affine_initialization
from oxford_untouched import load_capacity
from oxford_outcome_heldout_v2 import rows

def main():
 torch.set_num_threads(2);cells=load_capacity();cut=.8396755456924438;tr=rows(cells,range(1,5),'short',cut,'train');va=rows(cells,(5,6),'short',cut,'tail');te=rows(cells,(7,8),'short',cut,'tail');o={}
 t=time.perf_counter();m=fit_latent_regime_pp(tr,va,seed=42,affine_selection=select_affine_initialization(tr,va),max_epochs=400,patience=80,separation_weight=.01,gate_weight=0.,width=32);fit=time.perf_counter()-t;t=time.perf_counter();predict_latent_regime(m,te['x']);inf=time.perf_counter()-t;o['pp']={'train_seconds':fit,'inference_seconds':inf,'parameters':sum(p.numel() for p in m.model.parameters())}
 t=time.perf_counter();r=make_pipeline(StandardScaler(),Ridge(alpha=.1)).fit(tr['x'],tr['y']);fit=time.perf_counter()-t;t=time.perf_counter();r.predict(te['x']);inf=time.perf_counter()-t;o['ridge']={'train_seconds':fit,'inference_seconds':inf}
 # Frozen selected Engression configuration from the one-shot result, applied to short causal inputs for timing parity.
 torch.manual_seed(42);t=time.perf_counter();e=engression(torch.tensor(tr['x']),torch.tensor(tr['y'][:,None]),num_layer=3,hidden_dim=64,noise_dim=32,beta=.5,lr=.005,num_epoches=500,batch_size=len(tr['y']),device='cpu',standardize=True,verbose=False);fit=time.perf_counter()-t;t=time.perf_counter();e.predict(torch.tensor(te['x']),target='mean',sample_size=100);inf=time.perf_counter()-t;o['engression']={'train_seconds':fit,'inference_seconds':inf,'parameters':sum(p.numel() for p in e.model.parameters()) if hasattr(e,'model') else None}
 z={'hardware':'Apple arm CPU; torch threads=2','n_train':len(tr['y']),'n_test':len(te['y']),'scope':'single seed, fixed configurations, excludes search and loading','models':o};d=R/'results/oxford_compute_profile_v1';d.mkdir(exist_ok=True);(d/'results.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
if __name__=='__main__':main()
