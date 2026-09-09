"""Train-only waveform encoder, frozen latent-affine PP and matched GRU.

Retrospective FEMTO development, historical corrected 5+1 split. No full test
trajectories or test labels are used for fitting or checkpoint selection.
"""
import copy
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_sensor_adapter_v2 import load,RAW,OFFICIAL
from femto_corrected_benchmark_v2 import TRAIN,VAL,TEST
from pp_extrapolation.model import equal_group_weights

OUT=ROOT/'results/femto_waveform_pp_v7'
CACHE=ROOT/'data/femto/waveform_v7.npy'
REFERENCE=False
PROGRESS=False


def decode(score,elapsed,s):
    # Finite logit safety range is common to training, validation and inference.
    return (elapsed+50.)*torch.exp(-score.clamp(-6.,6.)) if PROGRESS else score.clamp(min=0)*s


def raw_cache(d):
    row_hash=hashlib.sha256(('\n'.join(f'{b}/{i}' for b,i in zip(d['bearing'],d['recording_index']))).encode()).hexdigest()
    manifest=CACHE.with_suffix('.json')
    if CACHE.exists():
        assert json.loads(manifest.read_text())['row_hash']==row_hash
        return np.load(CACHE,mmap_mode='r')
    a=np.lib.format.open_memmap(CACHE,mode='w+',dtype='float32',shape=(len(d['bearing']),2,2560))
    for j,(b,ix,role) in enumerate(zip(d['bearing'],d['recording_index'],d['role'])):
        parent='Learning_set' if role=='learn' else 'Test_set'
        p=RAW/parent/b/f'acc_{ix:05d}.csv'
        v=np.loadtxt(p,delimiter=',',usecols=(4,5),dtype=np.float32)
        assert v.shape==(2560,2)
        a[j]=v.T
        if j%500==0:print('raw',j,len(a),flush=True)
    a.flush();manifest.write_text(json.dumps({'row_hash':row_hash,'columns':[4,5],'shape':list(a.shape)}))
    return a


class WaveEncoder(nn.Module):
    def __init__(self,context_dim=6):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(2,8,17,stride=8,padding=8),nn.GELU(),
            nn.Conv1d(8,16,9,stride=4,padding=4),nn.GELU(),
            nn.Conv1d(16,16,5,stride=2,padding=2),nn.GELU())
        self.embed=nn.Sequential(nn.Linear(32+context_dim,24),nn.Tanh())
        self.head=nn.Linear(24,1)
    def forward(self,w,c):
        z=self.conv(w)
        h=self.embed(torch.cat([z.mean(-1),z.amax(-1),c],1))
        return self.head(h).squeeze(1),h


def train_encoder(raw,d,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tri=np.flatnonzero(np.isin(d['unit'],list(TRAIN)))
    vai=np.flatnonzero(np.isin(d['unit'],list(VAL)))
    # Shape/amplitude separation preserves physical amplitude as explicit input.
    amp=np.sqrt(np.mean(np.asarray(raw)**2,axis=2)+1e-8)
    c=np.c_[np.log(amp),np.eye(3)[d['condition'].astype(int)-1],np.log1p(d['elapsed_s'])/10].astype('float32')
    if REFERENCE:
        reference=np.zeros_like(amp)
        for bearing in np.unique(d['bearing']):
            ix=np.flatnonzero(d['bearing']==bearing);ix=ix[np.argsort(d['recording_index'][ix])]
            for j,k in enumerate(ix):reference[k]=np.median(np.log(amp[ix[:min(j+1,10)]]),axis=0)
        c=np.c_[c,np.log(amp)-reference,reference].astype('float32')
        vai=vai[np.unique(np.round((len(vai)-1)*np.array([.5,.6,.7,.8,.9])).astype(int))]
    mu=c[tri].mean(0);sd=np.maximum(c[tri].std(0),.1)
    cc=torch.tensor((c-mu)/sd)
    wave=torch.tensor(np.asarray(raw)/amp[:,:,None])
    s=float(d['y'][tri].max());y=torch.tensor(d['y'][tri]/s)
    elapsed=torch.tensor(d['elapsed_s'])
    progress=(elapsed[tri]+50.)/(elapsed[tri]+50.+torch.tensor(d['y'][tri]))
    weights=torch.tensor(equal_group_weights(d['bearing'][tri]),dtype=torch.float32)
    model=WaveEncoder(c.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    best=float('inf');ep=0;state=None;curve=[]
    for epoch in range(151):
        total=0.
        if epoch:
            model.train();order=rng.permutation(len(tri))
            for start in range(0,len(tri),128):
                ix=order[start:start+128];p,_=model(wave[tri[ix]],cc[tri[ix]])
                loss=(weights[ix]*((torch.sigmoid(p)-progress[ix]).square() if PROGRESS else (p-y[ix]).square())).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
                total+=float(loss.detach())*len(ix)/len(tri)
        model.eval()
        with torch.no_grad():
            p=decode(model(wave[vai],cc[vai])[0],elapsed[vai],s).numpy()
            v=float(np.mean((p-d['y'][vai])**2))
        curve.append([epoch,total if epoch else None,v])
        if v<best:best=v;ep=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-ep>30:break
    model.load_state_dict(state);model.eval();emb=[];base=[]
    with torch.no_grad():
        for start in range(0,len(wave),128):
            p,h=model(wave[start:start+128],cc[start:start+128]);emb.append(h.numpy());base.append(p.numpy())
    torch.save({'state':state,'context_mean':mu,'context_scale':sd,'target_scale':s},OUT/f'encoder_{seed}.pt')
    return np.concatenate(emb),np.concatenate(base),s,{'epoch':ep,'val_mse':best,'curve':curve}


def rows(emb,base,d,units,endpoint=False):
    xx=[];bb=[];yy=[];gg=[];tt=[]
    for unit in sorted(units):
        ix=np.flatnonzero(d['unit']==unit);ix=ix[np.argsort(d['recording_index'][ix])]
        values=np.c_[emb[ix],base[ix],np.log1p(d['elapsed_s'][ix])/10].astype('float32')
        for j in ([len(ix)-1] if endpoint else range(len(ix))):
            take=np.maximum(np.arange(j-15,j+1),0)
            xx.append(values[take]);bb.append(base[ix[j]]);yy.append(d['y'][ix[j]]);gg.append(d['bearing'][ix[j]])
            tt.append(d['elapsed_s'][ix[j]])
    return {'x':np.asarray(xx),'base':np.asarray(bb,dtype='float32'),'y':np.asarray(yy,dtype='float32'),'groups':np.asarray(gg),'elapsed':np.asarray(tt,dtype='float32')}


class TemporalHead(nn.Module):
    def __init__(self,mode,bound):
        super().__init__();self.mode=mode;self.bound=bound
        self.gru=nn.GRU(26,16,batch_first=True);self.head=nn.Linear(16,1)
        if mode=='pp':nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,b):
        _,h=self.gru(x);r=self.head(h[-1]).squeeze(1)
        return b+self.bound*torch.tanh(r/self.bound) if self.mode=='pp' else r


def fit_head(tr,va,s,seed,mode,bound):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mu=tr['x'][:,-1].mean(0);sd=np.maximum(tr['x'][:,-1].std(0),.05)
    tx=torch.tensor((tr['x']-mu)/sd);vx=torch.tensor((va['x']-mu)/sd)
    tb=torch.tensor(tr['base']);vb=torch.tensor(va['base']);y=torch.tensor(tr['y']/s)
    elapsed=torch.tensor(tr['elapsed']);vt=torch.tensor(va['elapsed'])
    progress=(elapsed+50.)/(elapsed+50.+torch.tensor(tr['y']))
    w=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32)
    model=TemporalHead(mode,bound);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.05)
    best=float('inf');ep=0;state=None;curve=[]
    for epoch in range(201):
        total=0.
        if epoch:
            model.train();order=rng.permutation(len(y))
            for start in range(0,len(y),256):
                ix=order[start:start+256];p=model(tx[ix],tb[ix]);loss=(w[ix]*((torch.sigmoid(p)-progress[ix]).square() if PROGRESS else (p-y[ix]).square())).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
                total+=float(loss.detach())*len(ix)/len(y)
        model.eval()
        with torch.no_grad():p=decode(model(vx,vb),vt,s).numpy();v=float(np.mean((p-va['y'])**2))
        curve.append([epoch,total if epoch else None,v])
        if v<best:best=v;ep=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-ep>40:break
    model.load_state_dict(state)
    return (model,mu,sd,s),{'epoch':ep,'val_mse':best,'curve':curve}


def predict(f,te):
    model,mu,sd,s=f
    with torch.no_grad():return decode(model(torch.tensor((te['x']-mu)/sd),torch.tensor(te['base'])),torch.tensor(te['elapsed']),s).numpy()


def calibration(kind,y,p):
    if kind=='identity':return (1.,0.)
    if kind=='scale':return (float(np.clip(np.dot(p,y)/max(np.dot(p,p),1e-9),.5,1.5)),0.)
    if kind=='bias':return (1.,float(np.mean(y-p)))
    slope,offset=np.polyfit(p,y,1)
    return (float(np.clip(slope,.5,1.5)),float(offset))


def choose_calibration(y,p):
    kinds=('identity','scale','bias','affine');scores={}
    for kind in kinds:
        q=[]
        for i in range(len(y)):
            mask=np.arange(len(y))!=i;a,b=calibration(kind,y[mask],p[mask]);q.append(max(a*p[i]+b,0.))
        scores[kind]=float(np.mean((np.asarray(q)-y)**2))
    kind=min(kinds,key=lambda k:scores[k]);return kind,calibration(kind,y,p),scores


def main():
    global OUT,REFERENCE,PROGRESS
    ap=argparse.ArgumentParser();ap.add_argument('--reference',action='store_true');ap.add_argument('--progress',action='store_true');args=ap.parse_args()
    PROGRESS=args.progress
    REFERENCE=args.reference or PROGRESS
    if REFERENCE:OUT=ROOT/'results/femto_waveform_pp_v8'
    if PROGRESS:OUT=ROOT/'results/femto_waveform_pp_v9'
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'runner_snapshot.py').write_text(Path(__file__).read_text())
    d=load();raw=raw_cache(d);models={};logs={};data={};search=[]
    for seed in range(42,47):
        emb,base,s,log=train_encoder(raw,d,seed)
        (OUT/f'encoder_curve_{seed}.json').write_text(json.dumps(log))
        tr=rows(emb,base,d,TRAIN);va=rows(emb,base,d,VAL);data[seed]=(emb,base,s)
        if REFERENCE:
            ix=np.unique(np.round((len(va['y'])-1)*np.array([.5,.6,.7,.8,.9])).astype(int))
            va={k:v[ix] for k,v in va.items()}
        print('ENCODER',seed,log['epoch'],log['val_mse'],flush=True)
        if seed==42:
            for bound in (.25,1.,3.):
                f,l=fit_head(tr,va,s,seed,'pp',bound)
                search.append({'bound':bound,'val_mse':l['val_mse']})
                models[('candidate',bound)]=f;logs[('candidate',bound)]=l
            choice=min(search,key=lambda a:a['val_mse'])['bound']
        for mode in ('pp','direct'):
            if seed==42 and mode=='pp':f,l=models[('candidate',choice)],logs[('candidate',choice)]
            else:f,l=fit_head(tr,va,s,seed,mode,choice)
            models[(mode,seed)]=f;logs[(mode,seed)]=l
            (OUT/f'{mode}_curve_{seed}.json').write_text(json.dumps(l))
            print('HEAD',seed,mode,l['epoch'],l['val_mse'],flush=True)
    (OUT/'selection_manifest.json').write_text(json.dumps({'status':'retrospective development','train':sorted(TRAIN),'validation':sorted(VAL),
        'test':sorted(TEST),'search':search,'bound':choice,'seeds':list(range(42,47)),'encoder':'five independent train-only CNN fits; shared within matched seed arms',
        'reference_context':REFERENCE,'validation_points':'50/60/70/80/90 percent observed prefixes' if REFERENCE else 'all validation rows',
        'progress_decoder':PROGRESS,'elapsed_offset_seconds':50. if PROGRESS else None,
        'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    results={}
    for mode in ('pp','direct','affine'):
        P=[];VP=[]
        for seed in range(42,47):
            emb,base,s=data[seed];te=rows(emb,base,d,TEST,True);va=rows(emb,base,d,VAL)
            if REFERENCE:
                ix=np.unique(np.round((len(va['y'])-1)*np.array([.5,.6,.7,.8,.9])).astype(int));va={k:v[ix] for k,v in va.items()}
            assert np.allclose(te['y'],[OFFICIAL[g] for g in te['groups']])
            P.append(decode(torch.tensor(te['base']),torch.tensor(te['elapsed']),s).numpy() if mode=='affine' else predict(models[(mode,seed)],te))
            VP.append(decode(torch.tensor(va['base']),torch.tensor(va['elapsed']),s).numpy() if mode=='affine' else predict(models[(mode,seed)],va))
        P=np.asarray(P);VP=np.asarray(VP);kind,(a,b),scores=choose_calibration(va['y'],VP.mean(0));calibrated=np.maximum(a*P+b,0.)
        results[mode]={'pooled_r2':float(r2_score(te['y'],P.mean(0))),
            'rmse':float(np.sqrt(np.mean((P.mean(0)-te['y'])**2))),'seed_r2':[float(r2_score(te['y'],p)) for p in P],
            'calibration':{'kind':kind,'slope':a,'offset':b,'loo_scores':scores},
            'calibrated_pooled_r2':float(r2_score(te['y'],calibrated.mean(0))),
            'calibrated_seed_r2':[float(r2_score(te['y'],p)) for p in calibrated]}
        np.savez_compressed(OUT/f'{mode}.npz',predictions=P,y=te['y'],groups=te['groups'],
            validation_prediction=VP,validation_y=va['y'],validation_groups=va['groups'])
        print('FINAL',mode,results[mode],flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
