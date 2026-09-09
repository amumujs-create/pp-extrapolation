"""Decoder-aligned, uncapped RUL PP for XJTU condition transfer."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from xjtu_spectrum_adapter_v2 import load
from xjtu_uncapped_temporal_pp_v2 import rows
from pp_extrapolation import regression_metrics
from pp_extrapolation.model import equal_group_weights
OUT=ROOT/'results/xjtu_aligned_decoder_pp_v4';SEEDS=range(42,47)

def inv_softplus(v):
 v=np.maximum(v,1e-5);return v+np.log(-np.expm1(-v))

class Net(nn.Module):
 def __init__(self,d,mode,bound,coef,bias):
  super().__init__();self.mode=mode;self.bound=bound;self.gru=nn.GRU(d,24,batch_first=True)
  self.affine=nn.Linear(d,1);self.direct=nn.Linear(24,1);self.residual=nn.Linear(24,1);self.capacity=nn.Linear(24,1)
  with torch.no_grad():self.affine.weight.copy_(torch.tensor(coef[None],dtype=torch.float32));self.affine.bias.fill_(float(bias))
  self.affine.requires_grad_(False);nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
  nn.init.zeros_(self.capacity.weight);nn.init.constant_(self.capacity.bias,-2.)
 def forward(self,x):
  _,h=self.gru(x);h=h[-1]
  if self.mode=='direct':return self.direct(h).squeeze(1)
  b=.15+(self.bound-.15)*torch.sigmoid(self.capacity(h).squeeze(1))
  return self.affine(x[:,-1]).squeeze(1)+b*torch.tanh(self.residual(h).squeeze(1)/b)

def fit(tr,va,c,seed):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mu=tr['x'][:,-1].mean(0);sd=np.maximum(tr['x'][:,-1].std(0),1e-6)
 tx=torch.tensor((tr['x']-mu)/sd);vx=torch.tensor((va['x']-mu)/sd);s=float(np.quantile(tr['y'],.9));target=tr['y']/s
 latent=inv_softplus(target);w=equal_group_weights(tr['groups']);ridge=Ridge(alpha=100.).fit(tx[:,-1].numpy(),latent,sample_weight=w)
 model=Net(tx.shape[-1],c['mode'],c['bound'],ridge.coef_,ridge.intercept_);params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=.05);yt=torch.tensor(tr['y']);tw=torch.tensor(w,dtype=torch.float32)
 best=float('inf');be=0;state=None;curve=[]
 def decode(raw):return s*torch.nn.functional.softplus(raw)
 for epoch in range(221):
  if epoch:
   model.train();order=rng.permutation(len(yt))
   for start in range(0,len(yt),512):
    ix=order[start:start+512];p=decode(model(tx[ix]));raw=((p-yt[ix])/s).square();log=(torch.log1p(p)-torch.log1p(yt[ix])).square()
    loss=(tw[ix]*(raw+c['log_weight']*log)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,2.);opt.step()
  model.eval()
  with torch.no_grad():p=decode(model(vx)).numpy();vl=float(np.mean((p-va['y'])**2))
  curve.append([epoch,vl])
  if vl<best:best=vl;be=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-be>40:break
 model.load_state_dict(state);return (model,mu,sd,s),{'epoch':be,'validation_mse':best,'curve':curve}

def predict(f,q):
 model,mu,sd,s=f
 with torch.no_grad():return (s*torch.nn.functional.softplus(model(torch.tensor((q['x']-mu)/sd)))).numpy()

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);raw=load();tr=rows(raw,'37.5Hz11kN','summary');va=rows(raw,'35Hz12kN','summary');te=rows(raw,'40Hz10kN','summary')
 configs=[{'mode':'direct','bound':1.,'lr':lr,'log_weight':lw} for lr in (.0005,.001) for lw in (0.,.1)]+[
  {'mode':'pp','bound':b,'lr':lr,'log_weight':lw} for b in (.5,1.5,3.) for lr in (.0005,.001) for lw in (0.,.1)]
 search=[];cache={}
 for i,c in enumerate(configs):
  f,l=fit(tr,va,c,42);cache[i]=(f,l);search.append({'index':i,'config':c,'epoch':l['epoch'],'validation_mse':l['validation_mse']});print('SCREEN',i,search[-1],flush=True)
 chosen={m:min([q for q in search if q['config']['mode']==m],key=lambda q:q['validation_mse']) for m in ('direct','pp')}
 (OUT/'selection_manifest.json').write_text(json.dumps({'status':'retrospective development; train/validation-only selection','selected':chosen,
  'search':search,'decoder':'s_train*softplus(raw), no train-RUL cap','source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 result={}
 for mode,choice in chosen.items():
  P=[];runs=[]
  for seed in SEEDS:
   f,l=cache[choice['index']] if seed==42 else fit(tr,va,choice['config'],seed);p=predict(f,te);P.append(p);runs.append({'seed':seed,'epoch':l['epoch'],'validation_mse':l['validation_mse'],'metrics':regression_metrics(te['y'],p,te['groups'])});print('SEED',mode,seed,runs[-1]['metrics']['pooled']['r2'],flush=True)
  P=np.asarray(P);result[mode]={'config':choice['config'],'runs':runs,'ensemble':regression_metrics(te['y'],P.mean(0),te['groups'])};np.savez_compressed(OUT/f'{mode}.npz',prediction=P,y=te['y'],groups=te['groups'])
  print('FINAL',mode,result[mode]['ensemble']['pooled'],flush=True)
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
