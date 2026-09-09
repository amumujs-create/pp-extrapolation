"""FEMTO health-threshold PP selected by Learning-bearing pseudo endpoints.

The prior predicts time until a causal health indicator reaches a failure
threshold learned from other complete Learning bearings.  No Test_set target is
used in selection. This is retrospective development because the dataset was
seen in earlier experiments.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from femto_sensor_adapter_v2 import load,LEARN,TEST,OFFICIAL

OUT=ROOT/'results/femto_health_threshold_pp_v10'
FRACTIONS=(.50,.60,.70,.80,.90,1.0)


def trajectory(d,name,feature,smooth):
    ix=np.flatnonzero(d['bearing']==name);ix=ix[np.argsort(d['recording_index'][ix])]
    raw=np.maximum(d['sensor'][ix,feature].astype(float),1e-12)
    base=float(np.median(raw[:min(10,len(raw))]))
    h=np.log(np.maximum(raw/base,1e-6))
    if smooth>1:h=median_filter(h,size=smooth,mode='nearest')
    # A running maximum is a damage-state prior, not a claim that vibration is
    # pointwise monotone. It prevents a transient drop from reversing damage.
    h=np.maximum.accumulate(h)
    return ix,d['elapsed_s'][ix].astype(float),h


def estimate(t,h,end,threshold,window,min_rate):
    start=max(0,end-window+1);tt=t[start:end+1];hh=h[start:end+1]
    if len(tt)<2:return 0.
    # Median pairwise slope is robust to isolated vibration bursts.
    dt=tt[:,None]-tt[None,:];dh=hh[:,None]-hh[None,:]
    mask=dt>0
    rate=max(float(np.median(dh[mask]/dt[mask])),min_rate)
    return max((threshold-h[end])/rate,0.)


def thresholds(d,feature,smooth,exclude=None,condition=None,quantile=.5):
    values=[]
    for name in LEARN:
        if name==exclude or (condition is not None and int(name[7])!=condition):continue
        _,_,h=trajectory(d,name,feature,smooth);values.append(h[-1])
    if not values:raise ValueError('no threshold source')
    return float(np.quantile(values,quantile))


def validation_score(d,c):
    errors=[];unit=[]
    for name in LEARN:
        ix,t,h=trajectory(d,name,c['feature'],c['smooth']);cond=int(name[7])
        th=thresholds(d,c['feature'],c['smooth'],exclude=name,condition=cond,quantile=c['quantile'])
        local=[]
        for fraction in FRACTIONS[:-1]:
            end=int(round((len(ix)-1)*fraction));p=estimate(t,h,end,th,c['window'],c['min_rate'])
            y=float(d['y'][ix[end]]);local.append((p-y)**2)
        errors.extend(local);unit.append(float(np.mean(local)))
    return float(np.mean(errors)),float(np.mean(unit))


def main():
    d=load();OUT.mkdir(parents=True,exist_ok=True)
    # sensor order per channel: rms,std,kurtosis,crest, three band magnitudes.
    configs=[]
    for feature in (0,1,4,5,6,7,8,11,12,13):
      for smooth in (1,3,7):
       for window in (8,16,32,64):
        for quantile in (.25,.5,.75):
         for min_rate in (1e-5,3e-5,1e-4):
          c={'feature':feature,'smooth':smooth,'window':window,'quantile':quantile,'min_rate':min_rate}
          mse,macro=validation_score(d,c);configs.append({**c,'validation_mse':mse,'validation_unit_macro_mse':macro})
    chosen=min(configs,key=lambda z:z['validation_unit_macro_mse'])
    manifest={'status':'retrospective development; selection uses Learning bearings only',
      'protocol':'leave-one-complete-bearing-out threshold; pseudo endpoints 50/60/70/80/90%; condition-matched threshold',
      'selected':chosen,'grid_size':len(configs),'grid':configs,
      'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    y=[];p=[];groups=[];rows=[]
    for name in TEST:
        ix,t,h=trajectory(d,name,chosen['feature'],chosen['smooth']);cond=int(name[7])
        th=thresholds(d,chosen['feature'],chosen['smooth'],condition=cond,quantile=chosen['quantile'])
        q=estimate(t,h,len(ix)-1,th,chosen['window'],chosen['min_rate'])
        truth=float(OFFICIAL[name]);y.append(truth);p.append(q);groups.append(name)
        rows.append({'bearing':name,'condition':cond,'truth':truth,'prediction':q,
          'health':float(h[-1]),'threshold':th,'margin':th-float(h[-1])})
    y=np.asarray(y);p=np.asarray(p);result={'selected':chosen,'pooled_r2':float(r2_score(y,p)),
      'rmse':float(np.sqrt(np.mean((y-p)**2))),'rows':rows,
      'claim':'deterministic health-threshold prior; not a neural PP win'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'predictions.npz',y=y,prediction=p,groups=np.asarray(groups))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
