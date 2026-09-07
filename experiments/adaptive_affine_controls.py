"""Matched tests of affine freezing and affine-anchored FT residual learning."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import extended_nn_benchmark as base
from pp_extrapolation import select_affine_initialization,regression_metrics

Original=base.Model
class AdaptiveModel(nn.Module):
 def __init__(self,kind,d,cfg,aff,z,vz):
  super().__init__();self.kind=kind
  if kind=='pp_adaptive':
   self.network=Original('pp',d,cfg,aff,z,vz);self.network.m.affine.requires_grad_(True)
  else:
   self.network=Original('ft_transformer',d,cfg,aff,z,vz)
   # Center the learned correction at its initial function, without changing
   # initialization of the Transformer or adding labels to the residual path.
   self.initial=copy.deepcopy(self.network).requires_grad_(False)
   self.affine=nn.Linear(d,1)
   with torch.no_grad():self.affine.weight.copy_(torch.tensor(aff['initialization'].weight)[None,:]);self.affine.bias.fill_(aff['initialization'].bias)
   self.affine.requires_grad_(kind=='ft_affine_adaptive')
 def forward(self,x):
  if self.kind=='pp_adaptive':return self.network(x)
  return self.affine(x).squeeze(1)+self.network(x)-self.initial(x)

def main():
 torch.set_num_threads(2);out=Path('results/adaptive_affine_controls_v1');out.mkdir(parents=True,exist_ok=True);parts=base.data();aff=select_affine_initialization(parts[0],parts[1]);te=parts[2];base.Model=AdaptiveModel
 for kind in ['pp_adaptive','ft_affine_frozen','ft_affine_adaptive']:
  if (out/f'{kind}.json').exists():continue
  cfgs=base.GRID if kind=='pp_adaptive' else [json.load(open('results/extended_nn_benchmark_v1/ft_transformer.json'))['selected_config']]
  candidates=[]
  if len(cfgs)>1:
   for cfg in cfgs:
    model,t,info=base.train(kind,cfg,42,parts,aff);candidates.append({'config':cfg,**info});print('SEARCH',kind,len(candidates),info['validation_mse'],flush=True)
   cfg=min(candidates,key=lambda x:x['validation_mse'])['config']
  else:cfg=cfgs[0]
  (out/f'{kind}_selection.json').write_text(json.dumps({'config':cfg,'candidates':candidates},indent=2));runs=[];pred=[]
  for seed in range(42,47):
   model,t,info=base.train(kind,cfg,seed,parts,aff)
   with torch.no_grad():raw=torch.cat([model(b) for b in t.split(512)]).numpy()*aff['target_scale']
   p=np.clip(raw,0,aff['target_scale']);pred.append(p);runs.append({'seed':seed,**info,'metrics':regression_metrics(te['y'],p,te['groups'])});np.savez_compressed(out/f'{kind}_{seed}.npz',raw=raw,prediction=p,y=te['y'],groups=te['groups'],ids=te['ids']);torch.save(model.state_dict(),out/f'{kind}_{seed}.pt');print('REFIT',kind,seed,flush=True)
  rs=[r['metrics']['pooled']['r2'] for r in runs];result={'status':'post-hoc development; architecture proposed after MATR test inspection','config':cfg,'candidates':candidates,'runs':runs,'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs)),'ensemble':regression_metrics(te['y'],np.mean(pred,0),te['groups'])};(out/f'{kind}.json').write_text(json.dumps(result,indent=2));print('DONE',kind,result['ensemble']['pooled']['r2'],flush=True)
if __name__=='__main__':main()
