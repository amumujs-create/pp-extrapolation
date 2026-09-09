"""Trajectory-matching PP for FEMTO using complete Learning bearings only."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from femto_sensor_adapter_v2 import load,LEARN,TEST,OFFICIAL
OUT=ROOT/'results/femto_trajectory_transport_pp_v11'
FEATURE_SETS={'rms':(0,7),'rms_std':(0,1,7,8),'bands':(0,4,5,6,7,11,12,13),'all':tuple(range(14))}

def curve(d,name,features,smooth):
    ix=np.flatnonzero(d['bearing']==name);ix=ix[np.argsort(d['recording_index'][ix])]
    z=np.sign(d['sensor'][ix][:,features])*np.log1p(np.abs(d['sensor'][ix][:,features]).astype(float))
    base=np.median(z[:min(10,len(z))],0);scale=np.maximum(np.median(np.abs(z[:min(20,len(z))]-base),0),.05)
    z=(z-base)/scale
    if smooth>1:
        q=np.empty_like(z)
        for i in range(len(z)):q[i]=np.median(z[max(0,i-smooth+1):i+1],0)
        z=q
    # current level + causal short/long changes
    f=[]
    for i in range(len(z)):
        f.append(np.r_[z[i],z[i]-z[max(0,i-4)],z[i]-z[max(0,i-16)]])
    return ix,np.asarray(f),d['elapsed_s'][ix].astype(float)

def signature(f,end,length):
    start=max(0,end-length+1);loc=np.linspace(start,end,8).round().astype(int)
    return f[loc].reshape(-1)

def predict_one(d,target,end,sources,c):
    feat=FEATURE_SETS[c['features']];tix,tf,tt=curve(d,target,feat,c['smooth']);query=signature(tf,end,c['history'])
    candidates=[]
    for source in sources:
        six,sf,st=curve(d,source,feat,c['smooth'])
        # Do not match the healthy initialization or terminal point.
        for j in range(max(7,c['history']//2),len(six)-1,c['step']):
            ref=signature(sf,j,c['history']);distance=float(np.mean((query-ref)**2))
            remaining=float(d['y'][six[j]])
            if c['scale']=='none':prediction=remaining
            elif c['scale']=='age':prediction=remaining*(tt[end]+50)/(st[j]+50)
            else:
                frac=(st[j]+50)/(st[j]+50+remaining);prediction=(tt[end]+50)*(1-frac)/max(frac,1e-3)
            candidates.append((distance,prediction,source,j))
    candidates.sort(key=lambda z:z[0]);near=candidates[:c['neighbors']]
    weight=1/(np.asarray([q[0] for q in near])+1e-4)
    pred=float(np.sum(weight*np.asarray([q[1] for q in near]))/weight.sum())
    return pred,near[0][0],near[0][2],near[0][3]

def validate(d,c):
    losses=[];units=[]
    for target in LEARN:
        ix=np.flatnonzero(d['bearing']==target);ix=ix[np.argsort(d['recording_index'][ix])]
        sources=[s for s in LEARN if s!=target and s[7]==target[7]]
        local=[]
        for fraction in (.5,.6,.7,.8,.9):
            end=int(round((len(ix)-1)*fraction));p,*_=predict_one(d,target,end,sources,c);local.append((p-float(d['y'][ix[end]]))**2)
        losses+=local;units.append(np.mean(local))
    return float(np.mean(losses)),float(np.mean(units))

def main():
    d=load();OUT.mkdir(parents=True,exist_ok=True);grid=[]
    # Bounded first-stage grid. The exhaustive Cartesian grid repeated the same
    # trajectory construction and was stopped before it produced a selection.
    for features in ('rms','rms_std','bands'):
      for smooth in (1,3):
       for history in (16,32):
        for step in (3,7):
         for neighbors in (1,3):
          for scale in ('none','age'):
           c=dict(features=features,smooth=smooth,history=history,step=step,neighbors=neighbors,scale=scale)
           mse,macro=validate(d,c);grid.append({**c,'validation_mse':mse,'validation_unit_macro_mse':macro})
    chosen=min(grid,key=lambda z:z['validation_unit_macro_mse'])
    (OUT/'selection_manifest.json').write_text(json.dumps({'status':'retrospective development; Test targets excluded from selection',
      'protocol':'same-condition leave-one-Learning-bearing-out trajectory alignment at five pseudo endpoints','grid_size':len(grid),
      'selected':chosen,'grid':grid,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    y=[];p=[];rows=[]
    for target in TEST:
        ix=np.flatnonzero(d['bearing']==target);ix=ix[np.argsort(d['recording_index'][ix])]
        sources=[s for s in LEARN if s[7]==target[7]];q,dist,source,j=predict_one(d,target,len(ix)-1,sources,chosen)
        truth=float(OFFICIAL[target]);y.append(truth);p.append(q);rows.append({'bearing':target,'truth':truth,'prediction':q,
          'nearest_source':source,'nearest_source_index':j,'distance':dist})
    y=np.asarray(y);p=np.asarray(p);result={'selected':chosen,'pooled_r2':float(r2_score(y,p)),
      'rmse':float(np.sqrt(np.mean((y-p)**2))),'rows':rows,'claim':'deterministic trajectory-transport PP; no neural residual'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=y,prediction=p,groups=np.asarray(TEST))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
