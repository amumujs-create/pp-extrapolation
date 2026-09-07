import os
os.environ['OMP_NUM_THREADS']='2'
import json,time,numpy as np,torch
from pathlib import Path
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
P=Path('results/extended_nn_benchmark_v1');d=np.load(P/'pfn_input.npz');torch.set_num_threads(2);rows=[];pred=[]
for seed in range(42,47):
 ix=np.random.default_rng(seed).choice(len(d['train_x']),min(1000,len(d['train_x'])),replace=False);start=time.monotonic()
 m=TabPFNRegressor.create_default_for_version(ModelVersion('v3'),device='cpu',n_estimators=1,random_state=seed,ignore_pretraining_limits=True)
 m.fit(d['train_x'][ix],d['train_y'][ix]);raw=m.predict(d['test_x']);p=np.clip(raw,0,max(d['train_y'].max(),1));pred.append(p);np.savez_compressed(P/f'pfn_{seed}.npz',prediction=p,raw=raw,y=d['test_y'],groups=d['groups'],train_indices=ix);rows.append({'seed':seed,'seconds':time.monotonic()-start});print('PFN',seed,flush=True)
(P/'pfn_runtime.json').write_text(json.dumps({'version':'8.0.7','checkpoint':'v3','training_rows':1000,'n_estimators':1,'status':'limited-context supplemental baseline; not full-data parity','runs':rows},indent=2))
