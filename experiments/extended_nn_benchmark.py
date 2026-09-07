"""Post-hoc neural benchmark: fixed 9 trials, validation-only selection, 5 refits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.benchmark_deps'))
import numpy as np
import torch
from torch import nn
from rtdl_revisiting_models import ResNet,FTTransformer
from pp_extrapolation import regression_metrics,select_affine_initialization
from pp_extrapolation.model import transform_features,equal_group_weights
from pp_extrapolation.regime_mixture import LatentRegimePPNet
from matr_2019_latent_confirmatory import load_cells,make_rows

OUT=Path('results/extended_nn_benchmark_v1')
KINDS=['pp','mlp','resnet','ft_transformer','moe','single','fixed','unconstrained','no_affine','gru','tcn','temporal_transformer']
GRID=[dict(width=w,depth=d,lr=lr,wd=wd) for (w,d) in [(16,1),(32,2),(64,2)] for lr,wd in [(2e-4,.1),(5e-4,2.),(1e-3,0.)]]
class Model(nn.Module):
 def __init__(self,kind,d,cfg,aff,z,vz):
  super().__init__();self.kind='pp' if kind=='pp_joint' else kind;kind=self.kind;w=cfg['width'];depth=cfg['depth']
  if kind in ('pp','single','fixed','unconstrained','no_affine'):
   direction=1. if vz[:,0].mean()>z[:,0].mean() else -1.
   self.m=LatentRegimePPNet(d,direction,float(np.quantile(direction*z[:,0],.8)),width=w)
   with torch.no_grad():
    self.m.affine.weight.copy_(torch.tensor(aff['initialization'].weight)[None,:]);self.m.affine.bias.fill_(aff['initialization'].bias)
    if kind=='no_affine':self.m.affine.weight.zero_();self.m.affine.bias.zero_()
   self.m.affine.requires_grad_(False)
   if kind=='unconstrained':self.slope=nn.Parameter(torch.tensor(.127))
  elif kind=='mlp':
   layers=[nn.Linear(d,w),nn.Tanh()]
   for _ in range(depth-1):layers += [nn.Linear(w,w),nn.Tanh()]
   self.m=nn.Sequential(*layers,nn.Linear(w,1))
  elif kind=='resnet':self.m=ResNet(d_in=d,d_out=1,n_blocks=depth,d_block=w,d_hidden=None,d_hidden_multiplier=2.,dropout1=0.,dropout2=0.)
  elif kind=='ft_transformer':self.m=FTTransformer(n_cont_features=d,cat_cardinalities=[],d_out=1,n_blocks=depth,d_block=w,attention_n_heads=4,attention_dropout=0.,ffn_d_hidden=None,ffn_d_hidden_multiplier=2.,ffn_dropout=0.,residual_dropout=0.)
  elif kind=='moe':
   self.g=nn.Sequential(nn.Linear(d,w),nn.Tanh(),nn.Linear(w,2));self.e=nn.ModuleList([nn.Sequential(nn.Linear(d,w),nn.Tanh(),nn.Linear(w,1)) for _ in range(2)])
  elif kind=='gru':self.m=nn.GRU(2,w,num_layers=depth,batch_first=True);self.head=nn.Linear(w,1)
  elif kind=='tcn':
   self.convs=nn.ModuleList([nn.Conv1d(2 if i==0 else w,w,3,dilation=2**i) for i in range(depth)]);self.head=nn.Linear(w,1)
  else:
   self.embed=nn.Linear(2,w);self.pos=nn.Parameter(torch.randn(1,8,w)*.01);layer=nn.TransformerEncoderLayer(w,4,w*2,dropout=0.,batch_first=True);self.m=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False);self.head=nn.Linear(w,1)
 def forward(self,x):
  k=self.kind
  if k in ('pp','single','fixed','unconstrained','no_affine'):
   m=self.m;q=m.direction*x[:,:1];h=torch.relu(q-m.knot);tails=[]
   for e in m.experts:
    raw=e(x);tails.append(.25*torch.tanh(raw[:,:1])+.25*torch.tanh(raw[:,1:])*h)
   if k=='single':g=torch.zeros_like(q)
   elif k=='fixed':g=torch.ones_like(q)*.5
   else:g=torch.sigmoid((self.slope if k=='unconstrained' else torch.nn.functional.softplus(m.gate_q))*q+m.gate_bias+m.gate_context(x[:,1:]))
   return (m.affine(x)+(1-g)*tails[0]+g*tails[1]).squeeze(-1)
  if k=='ft_transformer':return self.m(x,None).squeeze(-1)
  if k=='moe':return (torch.softmax(self.g(x),-1)*torch.cat([e(x) for e in self.e],-1)).sum(-1)
  if k=='gru':return self.head(self.m(x)[0][:,-1]).squeeze(-1)
  if k=='tcn':
   v=x.transpose(1,2)
   for c in self.convs:v=torch.relu(c(torch.nn.functional.pad(v,(2*c.dilation[0],0))))
   return self.head(v[:,:,-1]).squeeze(-1)
  if k=='temporal_transformer':return self.head(self.m(self.embed(x)+self.pos)[:,-1]).squeeze(-1)
  return self.m(x).squeeze(-1)

def data():
 audit=json.load(open('results/matr_2019_latent_confirmatory/results.json'))['pretest'];cs={c['index']:c for c in load_cells()[0]};parts=[]
 for name in ('train','validation','test'):
  cells=[cs[i] for i in audit['split_indices'][name]];p=make_rows(cells,audit['actual_boundary'],train=name=='train');seq=[];ids=[]
  for c in cells:
   q=c['q'];r=np.r_[0.,np.maximum(q[:-1]-q[1:],0.)]
   for end in range(7,len(q)):
    if (q[end]>audit['actual_boundary']) != (name=='train'):continue
    seq.append(np.stack([q[end-7:end+1],r[end-7:end+1]],axis=-1));ids.append(f"c{c['index']}:{end}")
  p['sequence']=np.array(seq,dtype=np.float32);p['ids']=np.array(ids);parts.append(p)
 return parts

def train(kind,cfg,seed,parts,aff):
 tr,va,te=parts;z=transform_features(tr['x'],aff['center'],aff['scale']);vz=transform_features(va['x'],aff['center'],aff['scale']);cap=aff['target_scale'];torch.manual_seed(seed);model=Model(kind,z.shape[1],cfg,aff,z,vz)
 temporal=kind in ('gru','tcn','temporal_transformer')
 if temporal:
  center=tr['sequence'].mean((0,1));scale=np.maximum(tr['sequence'].std((0,1)),1e-6);arrays=[(p['sequence']-center)/scale for p in parts]
 else:arrays=[transform_features(p['x'],aff['center'],aff['scale']) for p in parts]
 x,v,t=[torch.tensor(a,dtype=torch.float32) for a in arrays];y=torch.tensor(tr['y']/cap,dtype=torch.float32);weights=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']);rng=np.random.default_rng(seed)
 def val():
  model.eval()
  with torch.no_grad():p=torch.cat([model(b) for b in v.split(512)]).numpy()*cap
  return float(np.mean((np.clip(p,0,cap)-va['y'])**2))
 best=val();be=0;state=copy.deepcopy(model.state_dict());started=time.monotonic()
 for epoch in range(1,151):
  model.train();ix=rng.permutation(len(x))
  for start in range(0,len(x),512):
   b=ix[start:start+512];loss=(weights[b]*(model(x[b])-y[b])**2).mean()
   if kind=='pp_joint':
    _,g,tails=model.m.components(x[b]);reg={.0002:0.,.0005:.001,.001:.01}[cfg['lr']];loss=loss+reg*((g.mean()-.5).square()+(g*(1-g)).mean())-reg*torch.clamp(((tails[0]-tails[1])**2).mean(),max=.05)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  score=val()
  if score<best-1e-10:best=score;be=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-be>=25:break
 model.load_state_dict(state);model.eval()
 # No test prediction during search; caller only uses it for selected refits.
 return model,t,{'validation_mse':best,'selected_epoch':be,'epochs':epoch,'seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in model.parameters())}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--models',default=','.join(KINDS));args=parser.parse_args();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);parts=data();tr,va,te=parts;aff=select_affine_initialization(tr,va)
 for kind in args.models.split(','):
  dest=OUT/(kind+'.json')
  if dest.exists():continue
  rows=[]
  for i,cfg in enumerate(GRID):
   m,t,info=train(kind,cfg,42,parts,aff);rows.append({'config':cfg,**info});print('SEARCH',kind,i,info['validation_mse'],flush=True)
   del m,t
  chosen=min(range(len(rows)),key=lambda i:rows[i]['validation_mse']);cfg=GRID[chosen];(OUT/(kind+'_selection.json')).write_text(json.dumps({'candidates':rows,'chosen':chosen},indent=2));runs=[];pred=[]
  for seed in range(42,47):
   model,t,info=train(kind,cfg,seed,parts,aff)
   with torch.no_grad():raw=torch.cat([model(b) for b in t.split(512)]).numpy()*aff['target_scale']
   p=np.clip(raw,0,aff['target_scale']);pred.append(p);runs.append({'seed':seed,**info,'metrics':regression_metrics(te['y'],p,te['groups'])});np.savez_compressed(OUT/f'{kind}_{seed}.npz',raw=raw,prediction=p,y=te['y'],groups=te['groups'],ids=te['ids']);torch.save({'state_dict':model.state_dict(),'config':cfg,'kind':kind,'center':aff['center'],'scale':aff['scale'],'target_scale':aff['target_scale']},OUT/f'{kind}_{seed}.pt');print('REFIT',kind,seed,flush=True)
  rs=[r['metrics']['pooled']['r2'] for r in runs];dest.write_text(json.dumps({'status':'post-hoc','selected_config':cfg,'search':rows,'runs':runs,'mean_r2':float(np.mean(rs)),'sd_r2':float(np.std(rs)),'ensemble':regression_metrics(te['y'],np.mean(pred,0),te['groups'])},indent=2));print('DONE',kind,flush=True)
if __name__=='__main__':main()
