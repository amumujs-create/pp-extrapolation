"""Inductive FEMTO PP with bearing-specific causal order/envelope features."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_sensor_adapter_v2 import load,FS,OFFICIAL
from femto_corrected_gru_v2 import part
from femto_corrected_benchmark_v2 import TRAIN,VAL,TEST
from femto_waveform_pp_v7 import raw_cache
from femto_affine_gru_recovery_v6 import fit,prediction
OUT=ROOT/'results/femto_order_feature_pp_v15';CACHE=ROOT/'data/femto/order_features_v15.npz';SEEDS=range(42,47)
RPM={1:1800.,2:1650.,3:1500.}

def entropy(power):
 q=power/max(float(power.sum()),1e-12);return float(-np.sum(q*np.log(q+1e-12))/np.log(len(q)))

def one_channel(x,rpm):
 x=x-x.mean();window=x*np.hanning(len(x));spec=np.abs(np.fft.rfft(window));power=spec**2;freq=np.fft.rfftfreq(len(x),1/FS)
 env=np.abs(hilbert(x));env=env-env.mean();es=np.abs(np.fft.rfft(env*np.hanning(len(env))));ep=es**2
 shaft=rpm/60.;result=[]
 rms=np.sqrt(np.mean(x*x));result += [np.log1p(rms),np.mean((x/(x.std()+1e-9))**4)-3,
   np.max(np.abs(x))/(rms+1e-9),entropy(power[1:]),entropy(ep[1:])]
 # Log-spaced broadband energy captures resonance migration.
 for lo,hi in zip((0,250,500,1000,2000,4000,8000),(250,500,1000,2000,4000,8000,12800)):
  m=(freq>=lo)&(freq<hi);result.append(np.log1p(float(power[m].mean()) if m.any() else 0))
 # Shaft-order and envelope-order peaks; width covers FFT resolution.
 width=max(FS/len(x),shaft*.12)
 for order in (1,2,3,4,5,6,8,10,12,16):
  m=np.abs(freq-order*shaft)<=width;result += [np.log1p(float(power[m].mean())),np.log1p(float(ep[m].mean()))]
 return result

def build(d):
 raw=raw_cache(d);features=[]
 for i in range(len(raw)):
  rpm=RPM[int(d['condition'][i])];features.append(one_channel(raw[i,0],rpm)+one_channel(raw[i,1],rpm))
 x=np.asarray(features,np.float32);np.savez_compressed(CACHE,features=x,row_hash=hashlib.sha256(('\n'.join(f'{b}/{i}' for b,i in zip(d['bearing'],d['recording_index']))).encode()).hexdigest());return x

def main():
 import torch;torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);d=load();x=build(d) if not CACHE.exists() else np.load(CACHE)['features']
 payload={**d,'sensor':x};tr=part(payload,TRAIN);va=part(payload,VAL);te=part(payload,TEST,True)
 assert np.allclose(te['y'],[OFFICIAL[g] for g in te['groups']])
 configs=[{'mode':'direct','bound':1.}]+[{'mode':'pp','bound':b} for b in (.25,1.,3.,6.)]
 search=[];cache={}
 for i,c in enumerate(configs):
  f,l=fit(tr,va,c,42);search.append({'index':i,'config':c,'epoch':l['epoch'],'validation_mse':l['mse']});cache[i]=(f,l);print('SCREEN',search[-1],flush=True)
 chosen={m:min([q for q in search if q['config']['mode']==m],key=lambda q:q['validation_mse']) for m in ('direct','pp')}
 (OUT/'selection_manifest.json').write_text(json.dumps({'status':'retrospective inductive development; no TTA','feature_contract':'train-independent signal transform; causal per-record order/envelope features',
  'split':{'train':sorted(TRAIN),'validation':sorted(VAL),'test':sorted(TEST)},'search':search,'selected':chosen,
  'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 result={}
 for mode,ch in chosen.items():
  P=[];runs=[]
  for seed in SEEDS:
   f,l=cache[ch['index']] if seed==42 else fit(tr,va,ch['config'],seed);p=prediction(f,te);P.append(p);runs.append({'seed':seed,'epoch':l['epoch'],'validation_mse':l['mse'],'test_r2':float(r2_score(te['y'],p))});print('SEED',mode,runs[-1],flush=True)
  P=np.asarray(P);result[mode]={'config':ch['config'],'runs':runs,'pooled_r2':float(r2_score(te['y'],P.mean(0))),
    'rmse':float(np.sqrt(np.mean((te['y']-P.mean(0))**2)))};np.savez_compressed(OUT/f'{mode}.npz',prediction=P,y=te['y'],groups=te['groups']);print('FINAL',mode,result[mode],flush=True)
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
