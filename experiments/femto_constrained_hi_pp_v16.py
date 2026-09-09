"""Source-trained constrained-health-indicator PP for inductive FEMTO RUL."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_sensor_adapter_v2 import load,LEARN,TEST,OFFICIAL
from femto_order_feature_pp_v15 import CACHE as ORDER_CACHE,build as build_order
from pp_extrapolation.model import equal_group_weights
OUT=ROOT/'results/femto_constrained_hi_pp_v16'

def make_features(d,kind):
 raw=d['sensor'] if kind=='summary' else (np.load(ORDER_CACHE)['features'] if ORDER_CACHE.exists() else build_order(d))
 x=np.zeros((len(raw),3+1+3*raw.shape[1]),np.float32)
 for name in np.unique(d['bearing']):
  ix=np.flatnonzero(d['bearing']==name);ix=ix[np.argsort(d['recording_index'][ix])];z=np.sign(raw[ix])*np.log1p(np.abs(raw[ix]))
  base=np.median(z[:min(10,len(z))],0);scale=np.maximum(np.median(np.abs(z[:min(20,len(z))]-base),0),.05);z=(z-base)/scale
  for j,k in enumerate(ix):x[k]=np.r_[np.eye(3)[int(d['condition'][k])-1],np.log1p(d['elapsed_s'][k])/10,z[j],z[j]-z[max(0,j-4)],z[j]-z[max(0,j-16)]]
 return x

def fit_hi(d,x,train,alpha):
 ix=np.flatnonzero(np.isin(d['bearing'],train));progress=d['elapsed_s'][ix]/np.maximum(d['elapsed_s'][ix]+d['y'][ix],1)
 mu=x[ix].mean(0);sd=np.maximum(x[ix].std(0),.05);model=Ridge(alpha=alpha).fit((x[ix]-mu)/sd,progress,sample_weight=equal_group_weights(d['bearing'][ix]));return model,mu,sd

def trajectory(d,x,model,mu,sd,name):
 ix=np.flatnonzero(d['bearing']==name);ix=ix[np.argsort(d['recording_index'][ix])];h=np.clip(model.predict((x[ix]-mu)/sd),0,.999)
 # Projection onto a nondecreasing damage path; no future value enters an earlier prediction.
 return ix,np.maximum.accumulate(h),d['elapsed_s'][ix].astype(float)

def rul(h,t,end,window,scale,min_rate):
 a=max(0,end-window+1);tt=t[a:end+1];hh=h[a:end+1]
 rate=float(np.polyfit(tt,hh,1)[0]) if len(tt)>1 else 0.;rate=max(rate,min_rate)
 return max(scale*(1-h[end])/rate,0.)

def main():
 d=load();OUT.mkdir(parents=True,exist_ok=True);grid=[];features={k:make_features(d,k) for k in ('summary','order')}
 for held in LEARN:
  train=[u for u in LEARN if u!=held];condition=[u for u in train if u[7]==held[7]] or train
  for kind,x in features.items():
   for alpha in (1.,10.,100.,1000.):
    model,mu,sd=fit_hi(d,x,train,alpha);ix,h,t=trajectory(d,x,model,mu,sd,held)
    for window in (8,16,32,64):
     for scale in (.5,1.,1.5,2.):
      for min_rate in (1e-5,3e-5,1e-4):
       losses=[]
       for frac in (.5,.6,.7,.8,.9):
        end=int(round((len(ix)-1)*frac));p=rul(h,t,end,window,scale,min_rate);losses.append((p-d['y'][ix[end]])**2)
       grid.append({'held':held,'kind':kind,'alpha':alpha,'window':window,'scale':scale,'min_rate':min_rate,'mse':float(np.mean(losses))})
 # Choose one common config by unit-equal mean OOF error.
 keys={(q['kind'],q['alpha'],q['window'],q['scale'],q['min_rate']) for q in grid};summary=[]
 for key in keys:
  vals=[q['mse'] for q in grid if (q['kind'],q['alpha'],q['window'],q['scale'],q['min_rate'])==key]
  summary.append({'kind':key[0],'alpha':key[1],'window':key[2],'scale':key[3],'min_rate':key[4],'unit_macro_mse':float(np.mean(vals))})
 selected=min(summary,key=lambda q:q['unit_macro_mse']);x=features[selected['kind']];model,mu,sd=fit_hi(d,x,LEARN,selected['alpha'])
 manifest={'status':'retrospective inductive development; no TTA','selection':'leave-one-complete-Learning-bearing-out pseudo endpoints','selected':selected,
  'grid':summary,'test_batch_statistics_used':False,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');y=[];p=[];rows=[]
 for name in TEST:
  ix,h,t=trajectory(d,x,model,mu,sd,name);q=rul(h,t,len(ix)-1,selected['window'],selected['scale'],selected['min_rate']);truth=float(OFFICIAL[name]);y.append(truth);p.append(q);rows.append({'bearing':name,'truth':truth,'prediction':q,'health':float(h[-1])})
 y=np.asarray(y);p=np.asarray(p);result={'model':'constrained HI boundary PP','pooled_r2':float(r2_score(y,p)),'rmse':float(np.sqrt(np.mean((y-p)**2))),'rows':rows}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=y,prediction=p,groups=np.asarray(TEST));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
