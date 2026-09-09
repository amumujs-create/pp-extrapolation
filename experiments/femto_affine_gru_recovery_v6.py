"""Matched GRU versus low-dimensional affine + GRU PP, historical 5+1 split."""
import copy, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_corrected_gru_v2 import part
from femto_corrected_benchmark_v2 import TRAIN,VAL,TEST
from femto_sensor_adapter_v2 import load
from pp_extrapolation.model import equal_group_weights
OUT=ROOT/'results/femto_affine_gru_recovery_v6'

class Network(nn.Module):
    def __init__(self,d,w,b,mode,bound):
        super().__init__(); self.mode=mode; self.bound=bound
        self.encoder=nn.GRU(d,16,batch_first=True)
        self.head=nn.Linear(16,1)
        self.affine=nn.Linear(2,1)
        with torch.no_grad():
            self.affine.weight.copy_(torch.tensor(w[None],dtype=torch.float32));self.affine.bias.fill_(float(b))
        self.affine.requires_grad_(False)
        if mode=='pp':
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        _,h=self.encoder(x);r=self.head(h[-1]).squeeze(-1)
        if self.mode=='pp':
            return self.affine(x[:,-1,:2]).squeeze(-1)+self.bound*torch.tanh(r/self.bound)
        return r

def fit(tr,va,c,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mu=tr['x'][:,-1].mean(0);sd=tr['x'][:,-1].std(0);sd=np.where(sd<1e-8,1,sd)
    s=max(float(tr['y'].max()),1.)
    tx=torch.tensor((tr['x']-mu)/sd);vx=torch.tensor((va['x']-mu)/sd)
    y=torch.tensor(tr['y']/s);w=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32)
    a=Ridge(alpha=100.).fit(tx[:,-1,:2].numpy(),y.numpy(),sample_weight=w.numpy())
    model=Network(tx.shape[-1],a.coef_,a.intercept_,c['mode'],c['bound'])
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.05)
    best=float('inf');state=None;epoch_best=0;curve=[]
    for epoch in range(301):
        if epoch:
            model.train();order=rng.permutation(len(y))
            for start in range(0,len(y),512):
                ix=order[start:start+512];p=model(tx[ix]);loss=(w[ix]*(p-y[ix]).square()).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        model.eval()
        with torch.no_grad():
            p=model(vx).clamp(min=0).numpy()*s;v=float(np.mean((p-va['y'])**2))
        curve.append([epoch,v])
        if v<best:
            best=v;epoch_best=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-epoch_best>60:break
    model.load_state_dict(state)
    return (model,mu,sd,s),{'epoch':epoch_best,'mse':best,'curve':curve}

def prediction(f,rows):
    model,mu,sd,s=f
    with torch.no_grad():return model(torch.tensor((rows['x']-mu)/sd)).clamp(min=0).numpy()*s

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    d=load();tr=part(d,TRAIN);va=part(d,VAL)
    configs=[{'mode':'direct','bound':1.}]+[{'mode':'pp','bound':b} for b in (.25,1.,3.)]
    search=[];cache={}
    for i,c in enumerate(configs):
        f,l=fit(tr,va,c,42);cache[i]=(f,l);search.append({'config':c,'epoch':l['epoch'],'mse':l['mse']})
        (OUT/f'curve_search{i}.json').write_text(json.dumps(l));print('SCREEN',i,search[-1],flush=True)
    chosen={m:min([i for i,c in enumerate(configs) if c['mode']==m],key=lambda i:search[i]['mse']) for m in ('direct','pp')}
    (OUT/'selection_manifest.json').write_text(json.dumps({'protocol':'historical corrected 5+1; 32-step GRU; lower-only output; train/val selection',
        'search':search,'chosen':chosen,'status':'retrospective development','source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    te=part(d,TEST,True);results={}
    for mode,i in chosen.items():
        preds=[];runs=[]
        for seed in range(42,47):
            f,l=cache[i] if seed==42 else fit(tr,va,configs[i],seed)
            p=prediction(f,te);preds.append(p);runs.append({'seed':seed,'epoch':l['epoch'],'val_mse':l['mse'],'r2':float(r2_score(te['y'],p))})
            (OUT/f'curve_{mode}_{seed}.json').write_text(json.dumps(l));print('SEED',mode,runs[-1],flush=True)
        P=np.asarray(preds);results[mode]={'runs':runs,'r2':float(r2_score(te['y'],P.mean(0))),'rmse':float(np.sqrt(np.mean((te['y']-P.mean(0))**2)))}
        np.savez_compressed(OUT/f'{mode}.npz',predictions=P,y=te['y'],groups=te['groups'])
        print('FINAL',mode,results[mode],flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
