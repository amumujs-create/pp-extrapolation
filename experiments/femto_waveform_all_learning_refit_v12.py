"""All-six Learning-bearing refit of the FEMTO waveform safety executor."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_sensor_adapter_v2 import load,LEARN,TEST,OFFICIAL
import femto_waveform_pp_v7 as wave
from pp_extrapolation.model import equal_group_weights
OUT=ROOT/'results/femto_waveform_all_learning_refit_v12';ENCODER_EPOCHS=2

def context(raw,d,train):
    amp=np.sqrt(np.mean(np.asarray(raw)**2,axis=2)+1e-8)
    reference=np.zeros_like(amp)
    for bearing in np.unique(d['bearing']):
        ix=np.flatnonzero(d['bearing']==bearing);ix=ix[np.argsort(d['recording_index'][ix])]
        for j,k in enumerate(ix):reference[k]=np.median(np.log(amp[ix[:min(j+1,10)]]),axis=0)
    c=np.c_[np.log(amp),np.eye(3)[d['condition'].astype(int)-1],np.log1p(d['elapsed_s'])/10,
            np.log(amp)-reference,reference].astype('float32')
    mu=c[train].mean(0);sd=np.maximum(c[train].std(0),.1)
    return amp,(c-mu)/sd,mu,sd

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);d=load();raw=wave.raw_cache(d)
    train=np.flatnonzero(np.isin(d['bearing'],LEARN));test=[]
    for name in TEST:
        ix=np.flatnonzero(d['bearing']==name);test.append(ix[np.argmax(d['recording_index'][ix])])
    test=np.asarray(test);assert np.allclose(d['y'][test],[OFFICIAL[n] for n in TEST])
    amp,c,mu,sd=context(raw,d,train);x=torch.tensor(np.asarray(raw)/amp[:,:,None]);c=torch.tensor(c)
    scale=float(d['y'][train].max());y=torch.tensor(d['y'][train]/scale);w=torch.tensor(equal_group_weights(d['bearing'][train]),dtype=torch.float32)
    preds=[];runs=[]
    for seed in range(42,47):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=wave.WaveEncoder(c.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05);curve=[]
        for epoch in range(1,ENCODER_EPOCHS+1):
            model.train();order=rng.permutation(len(train));total=0.
            for start in range(0,len(train),128):
                q=order[start:start+128];p,_=model(x[train[q]],c[train[q]]);loss=(w[q]*(p-y[q]).square()).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();total+=float(loss.detach())*len(q)/len(train)
            curve.append([epoch,total])
        model.eval()
        with torch.no_grad():p=model(x[test],c[test])[0].clamp(min=0).numpy()*scale
        preds.append(p);runs.append({'seed':seed,'train_curve':curve,'test_r2':float(r2_score(d['y'][test],p))});print(seed,runs[-1],flush=True)
        torch.save({'state':model.state_dict(),'context_mean':mu,'context_scale':sd,'target_scale':scale},OUT/f'seed{seed}.pt')
    P=np.asarray(preds);result={'status':'retrospective development; deterministic epoch borrowed from prior validation experiment',
      'route':'no admissible FEMTO prior: all-six Learning waveform NN safety executor','encoder_epochs':ENCODER_EPOCHS,'runs':runs,
      'pooled_r2':float(r2_score(d['y'][test],P.mean(0))),'rmse':float(np.sqrt(np.mean((d['y'][test]-P.mean(0))**2))),
      'limitations':['not a PP-prior performance win','five-seed prediction mean; individual deployment remains unstable','test dataset was seen in prior development'],
      'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',predictions=P,y=d['y'][test],groups=d['bearing'][test]);print('FINAL',result,flush=True)
if __name__=='__main__':main()
