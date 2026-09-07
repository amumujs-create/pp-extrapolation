"""Post-hoc equal-trial controls; no changes to historical confirmation."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from pp_extrapolation import regression_metrics,select_affine_initialization
from pp_extrapolation.model import transform_features,equal_group_weights
from pp_extrapolation.regime_mixture import LatentRegimePPNet
from plain_mlp_ablation import PlainMLP
from matr_2019_latent_confirmatory import load_cells,make_rows
OUT=Path('results/submission_controls_v1');GRID=[(lr,wd) for lr in (0.0002,0.0005,0.001) for wd in (0.,.1,2.)]
class Control(nn.Module):
 def __init__(self,d,direction,knot,kind):
  super().__init__();self.net=LatentRegimePPNet(d,direction,knot);self.kind=kind
 def forward(self,x):
  net=self.net;q=net.direction*x[:,:1];h=torch.relu(q-net.knot)
  tails=[]
  for expert in net.experts:
   raw=expert(x);tails.append(.25*torch.tanh(raw[:,:1])+.25*torch.tanh(raw[:,1:])*h)
  correction=tails[0] if self.kind=='single' else (tails[0]+tails[1])/2
  return (net.affine(x)+correction).squeeze(1)
def fit(tr,va,aff,seed,kind,lr,wd):
 torch.manual_seed(seed);z=transform_features(tr['x'],aff['center'],aff['scale']);vz=transform_features(va['x'],aff['center'],aff['scale']);cap=aff['target_scale'];direction=1 if vz[:,0].mean()>z[:,0].mean() else -1
 model=PlainMLP(z.shape[1]) if kind=='plain' else Control(z.shape[1],direction,float(np.quantile(direction*z[:,0],.8)),kind)
 if kind!='plain':
  with torch.no_grad():model.net.affine.weight.copy_(torch.tensor(aff['initialization'].weight)[None,:]);model.net.affine.bias.fill_(aff['initialization'].bias)
  model.net.affine.requires_grad_(False)
 x=torch.tensor(z);v=torch.tensor(vz);y=torch.tensor(tr['y']/cap);w=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32);rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
 def val():
  with torch.no_grad():return float(np.mean((np.clip(model(v).numpy()*cap,0,cap)-va['y'])**2))
 best=val();epochbest=0;state=copy.deepcopy(model.state_dict())
 for epoch in range(1,301):
  order=rng.permutation(len(x))
  for start in range(0,len(x),512):
   ix=order[start:start+512];loss=(w[ix]*(model(x[ix])-y[ix])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  score=val()
  if score<best-1e-10:best=score;epochbest=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-epochbest>70:break
 model.load_state_dict(state);return best,model,epochbest

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 old=json.load(open('results/matr_2019_latent_confirmatory/results.json'));audit=old['pretest'];cells={c['index']:c for c in load_cells()[0]};b=audit['actual_boundary'];parts=[]
 for name in ('train','validation','test'):parts.append(make_rows([cells[i] for i in audit['split_indices'][name]],b,train=name=='train'))
 tr,va,te=parts;aff=select_affine_initialization(tr,va);testz=torch.tensor(transform_features(te['x'],aff['center'],aff['scale']));results={'status':'post-hoc equal candidate count, not equal runtime','grid':GRID,'datasets':{}};saved={'y':te['y'],'groups':te['groups']}
 for kind in ('plain','single','fixed'):
  predictions=[];runs=[];start=time.monotonic()
  for seed in range(42,47):
   best=None;candidates=[]
   for lr,wd in GRID:
    score,model,epoch=fit(tr,va,aff,seed,kind,lr,wd);candidates.append({'lr':lr,'wd':wd,'validation_mse':score,'epoch':epoch})
    if best is None or score<best[0]:best=(score,model,lr,wd,epoch)
   with torch.no_grad():raw=best[1](testz).numpy()*aff['target_scale']
   p=np.clip(raw,0,aff['target_scale']);predictions.append(p);saved[f'{kind}_{seed}_raw']=raw;runs.append({'seed':seed,'lr':best[2],'wd':best[3],'epoch':best[4],'candidates':candidates,'metrics':regression_metrics(te['y'],p,te['groups'])});print(kind,seed,runs[-1]['metrics']['pooled']['r2'],flush=True)
  rs=[r['metrics']['pooled']['r2'] for r in runs];results['datasets'][kind]={'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs)),'seconds':time.monotonic()-start,'ensemble':regression_metrics(te['y'],np.mean(predictions,axis=0),te['groups']),'runs':runs};(OUT/'results.json').write_text(json.dumps(results,indent=2))
 # A nonlinear tabular control, validation-selected with nine trials.
 candidates=[]
 for leaves in (7,15,31):
  for l2 in (0.,1.,10.):
   m=HistGradientBoostingRegressor(max_leaf_nodes=leaves,l2_regularization=l2,max_iter=300,early_stopping=False,random_state=42).fit(tr['x'],tr['y'],sample_weight=equal_group_weights(tr['groups']));score=np.mean((np.clip(m.predict(va['x']),0,aff['target_scale'])-va['y'])**2);candidates.append((score,leaves,l2,m))
 best=min(candidates,key=lambda t:t[:3]);p=np.clip(best[3].predict(te['x']),0,aff['target_scale']);saved['histgb']=p;results['datasets']['histgb']={'ensemble':regression_metrics(te['y'],p,te['groups']),'leaves':best[1],'l2':best[2]};(OUT/'results.json').write_text(json.dumps(results,indent=2));np.savez_compressed(OUT/'predictions.npz',**saved)
if __name__=='__main__':main()
