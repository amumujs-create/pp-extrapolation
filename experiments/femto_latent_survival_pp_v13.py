"""Latent failure-mode total-life PP for corrected FEMTO data.

The analytic path is a condition-specific log-total-life prior. A causal GRU
selects bounded latent failure-mode corrections. RUL is decoded as
softplus(T_hat - elapsed), so the model estimates a lifetime scale rather than
regressing endpoint RUL directly. Retrospective development, historical 5+1.
"""
from __future__ import annotations
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_sensor_adapter_v2 import load,OFFICIAL
from femto_spectrum_adapter_v3 import load as load_spectrum
from femto_corrected_benchmark_v2 import TRAIN,VAL,TEST
from pp_extrapolation.model import equal_group_weights
OUT=ROOT/'results/femto_latent_survival_pp_v13'
WINDOW=32;SEEDS=range(42,47)


def causal_rows(d,kind,units,endpoint=False):
    xs=[];ys=[];total=[];elapsed=[];groups=[];conditions=[]
    source=d['sensor'] if kind=='summary' else np.c_[d['sensor'],load_spectrum()['spectrum']]
    for unit in sorted(units):
        ix=np.flatnonzero(d['unit']==unit);ix=ix[np.argsort(d['recording_index'][ix])]
        raw=source[ix].astype(float);base=np.median(raw[:min(10,len(raw))],0)
        # Signed logarithm handles powers and signed summary statistics together.
        z=np.sign(raw)*np.log1p(np.abs(raw));b=np.sign(base)*np.log1p(np.abs(base))
        scale=np.maximum(np.median(np.abs(z[:min(20,len(z))]-b),0),.05)
        norm=(z-b)/scale;t=d['elapsed_s'][ix].astype(float)
        current=[]
        for j in range(len(ix)):
            current.append(np.r_[np.eye(3)[int(d['condition'][ix[j]])-1],np.log1p(t[j])/10,
                norm[j],norm[j]-norm[max(0,j-4)],norm[j]-norm[max(0,j-16)],
                min(j+1,WINDOW)/WINDOW])
        current=np.asarray(current,np.float32);chosen=[len(ix)-1] if endpoint else range(len(ix))
        for j in chosen:
            q=np.maximum(np.arange(j-WINDOW+1,j+1),0);xs.append(current[q]);ys.append(d['y'][ix[j]])
            elapsed.append(t[j]);total.append(t[j]+d['y'][ix[j]]);groups.append(d['bearing'][ix[j]])
            conditions.append(int(d['condition'][ix[j]])-1)
    return {'x':np.asarray(xs),'y':np.asarray(ys,np.float32),'total':np.asarray(total,np.float32),
      'elapsed':np.asarray(elapsed,np.float32),'groups':np.asarray(groups),'condition':np.asarray(conditions)}


class SurvivalNet(nn.Module):
    def __init__(self,dim,mode,modes,bound,prior):
        super().__init__();self.mode=mode;self.modes=modes;self.bound=bound
        self.encoder=nn.GRU(dim,32,batch_first=True)
        self.gate=nn.Linear(32,modes);self.expert=nn.Linear(32,modes);self.direct=nn.Linear(32,1)
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32))
        nn.init.zeros_(self.expert.weight);nn.init.zeros_(self.expert.bias)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
    def forward(self,x,condition,elapsed):
        _,h=self.encoder(x);h=h[-1]
        if self.mode=='direct':loglife=self.direct(h).squeeze(1)
        else:
            weights=torch.softmax(self.gate(h),1);correction=self.bound*torch.tanh(self.expert(h)/self.bound)
            loglife=self.prior[condition]+torch.sum(weights*correction,1)
        life=torch.exp(loglife.clamp(0,12));rul=torch.nn.functional.softplus((life-elapsed)/100.)*100.
        return rul,loglife,torch.softmax(self.gate(h),1)


def prior_fit(rows):
    # Equal-unit weighting prevents the longest trajectory fixing the prior.
    onehot=np.eye(3)[rows['condition']];w=equal_group_weights(rows['groups'])
    model=Ridge(alpha=10.).fit(onehot,np.log(np.maximum(rows['total'],1)),sample_weight=w)
    return model.predict(np.eye(3)).astype(np.float32)


def fit(tr,va,c,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mu=tr['x'][:,-1].mean(0);sd=np.maximum(tr['x'][:,-1].std(0),.05)
    def tensors(q):return (torch.tensor((q['x']-mu)/sd),torch.tensor(q['condition']),torch.tensor(q['elapsed']))
    tx,tc,tt=tensors(tr);vx,vc,vt=tensors(va);y=torch.tensor(tr['y']);logt=torch.tensor(np.log(np.maximum(tr['total'],1)))
    w=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32);scale=max(float(np.median(tr['total'])),1.)
    prior=prior_fit(tr);model=SurvivalNet(tx.shape[-1],c['mode'],c['modes'],c['bound'],prior)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.05)
    best=float('inf');best_epoch=0;state=None;curve=[]
    for epoch in range(181):
        if epoch:
            model.train();order=rng.permutation(len(y))
            for start in range(0,len(y),256):
                ix=order[start:start+256];p,lt,g=model(tx[ix],tc[ix],tt[ix])
                raw=((p-y[ix])/scale).square();life=(lt-logt[ix]).square()
                # Same-unit total-life predictions should not jump with prefix length.
                loss=(w[ix]*(raw+.1*life)).mean()
                if c['modes']>1:
                    balance=(g.mean(0)-1/c['modes']).square().mean();confidence=-(g*torch.log(g+1e-8)).sum(1).mean()
                    loss=loss+.01*balance+c['entropy']*confidence
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        model.eval()
        with torch.no_grad():p,_,g=model(vx,vc,vt);vl=float(np.mean((p.numpy()-va['y'])**2))
        curve.append([epoch,vl,float(g.max(1).values.mean())])
        if vl<best:best=vl;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>30:break
    model.load_state_dict(state);return (model,mu,sd),{'epoch':best_epoch,'validation_mse':best,'curve':curve,'prior_loglife':prior.tolist()}


def predict(f,rows):
    model,mu,sd=f
    with torch.no_grad():p,lt,g=model(torch.tensor((rows['x']-mu)/sd),torch.tensor(rows['condition']),torch.tensor(rows['elapsed']))
    return p.numpy(),lt.numpy(),g.numpy()


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);d=load();prepared={}
    for kind in ('summary','spectrum'):
        prepared[kind]=(causal_rows(d,kind,TRAIN),causal_rows(d,kind,VAL),causal_rows(d,kind,TEST,True))
    configs=[]
    for kind in ('summary','spectrum'):
        configs.append({'kind':kind,'mode':'direct','modes':1,'bound':1.,'lr':.001,'entropy':0.})
        configs.append({'kind':kind,'mode':'pp','modes':1,'bound':.5,'lr':.001,'entropy':0.})
        for bound in (.5,1.5):
            configs.append({'kind':kind,'mode':'pp','modes':3,'bound':bound,'lr':.001,'entropy':.001})
    search=[];cache={}
    for i,c in enumerate(configs):
        tr,va,_=prepared[c['kind']];f,l=fit(tr,va,c,42);cache[i]=(f,l)
        search.append({'index':i,'config':c,'epoch':l['epoch'],'validation_mse':l['validation_mse']});print('SCREEN',i,search[-1],flush=True)
    chosen={m:min([r for r in search if r['config']['mode']==m],key=lambda r:r['validation_mse']) for m in ('direct','pp')}
    (OUT/'selection_manifest.json').write_text(json.dumps({'status':'retrospective development; train/validation-only selection',
      'split':{'train':sorted(TRAIN),'validation':sorted(VAL),'test':sorted(TEST)},'objective':'Bearing3_2 all-prefix MSE',
      'search':search,'chosen':chosen,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    results={}
    for mode,choice in chosen.items():
        c=choice['config'];tr,va,te=prepared[c['kind']];P=[];runs=[];G=[]
        for seed in SEEDS:
            f,l=cache[choice['index']] if seed==42 else fit(tr,va,c,seed);p,lt,g=predict(f,te);P.append(p);G.append(g)
            runs.append({'seed':seed,'epoch':l['epoch'],'validation_mse':l['validation_mse'],'test_r2':float(r2_score(te['y'],p))})
            (OUT/f'{mode}_curve_{seed}.json').write_text(json.dumps(l));print('SEED',mode,runs[-1],flush=True)
        P=np.asarray(P);G=np.asarray(G);result_row={'config':c,'runs':runs,'pooled_r2':float(r2_score(te['y'],P.mean(0))),
          'rmse':float(np.sqrt(np.mean((te['y']-P.mean(0))**2))),'mean_gate_confidence':float(G.max(-1).mean())}
        results[mode]=result_row;np.savez_compressed(OUT/f'{mode}.npz',predictions=P,y=te['y'],groups=te['groups'],gates=G)
        print('FINAL',mode,result_row,flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
