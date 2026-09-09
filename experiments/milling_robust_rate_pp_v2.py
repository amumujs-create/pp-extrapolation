"""Validation-selected robust causal-rate boundary PP for NASA milling."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import theilslopes
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from nasa_milling_causal import MAT,BOUNDARY,TRAIN_CASES,VALIDATION_CASES,SOURCE_CASES
from pp_extrapolation import regression_metrics
OUT=ROOT/'results/milling_robust_rate_pp_v2'

def cases():
 out={}
 for r in loadmat(MAT,simplify_cells=True)['mill']:
  if np.isfinite(float(r['VB'])):out.setdefault(int(r['case']),[]).append((float(r['time']),float(r['VB'])))
 return {k:sorted(v) for k,v in out.items()}

def rows(data,units,window,method,offset):
 y=[];p=[];g=[];rate=[]
 for unit in units:
  obs=data.get(unit,[]);cross=[t for t,w in obs if w>=BOUNDARY]
  if not cross:continue
  eol=cross[0];prefix=[q for q in obs if q[0]<eol];t=np.asarray([q[0] for q in prefix]);w=np.asarray([q[1] for q in prefix])
  # Keep the same causal eligibility contract as the frozen reconstruction.
  n=np.arange(1,len(t)+1);den=n*np.cumsum(t*t)-np.cumsum(t)**2;r=np.full(len(t),np.nan);ok=den>0
  r[ok]=(n[ok]*np.cumsum(t*w)[ok]-np.cumsum(t)[ok]*np.cumsum(w)[ok])/den[ok]
  eligible=np.flatnonzero((BOUNDARY-w>0)&(n>=2)&(r>0))
  for pos in range(1,len(eligible)):
   i=eligible[pos];a=max(0,i-window+1);tt=t[a:i+1];ww=w[a:i+1]
   slope=float(theilslopes(ww,tt)[0]) if method=='theil' and len(tt)>2 else float(np.polyfit(tt,ww,1)[0])
   slope=max(slope,1e-6);y.append(eol-t[i]);p.append(max((BOUNDARY+offset-w[i])/slope,0));g.append(unit);rate.append(slope)
 return np.asarray(y),np.asarray(p),np.asarray(g),np.asarray(rate)

def main():
 data=cases();grid=[]
 for window in (2,3,5,8,1000):
  for method in ('ols','theil'):
   for offset in np.arange(-.05,.101,.01):
    y,p,g,_=rows(data,VALIDATION_CASES,window,method,float(offset));grid.append({'window':window,'method':method,'offset':float(offset),
      'validation_mae':float(np.mean(np.abs(y-p))),'validation_rmse':float(np.sqrt(np.mean((y-p)**2)))})
 selected=min(grid,key=lambda q:(q['validation_mae'],q['validation_rmse'],abs(q['offset'])));y,p,g,r=rows(data,SOURCE_CASES,selected['window'],selected['method'],selected['offset'])
 result={'status':'retrospective development; validation-only robust-rate selection','selected':selected,'grid':grid,
  'test':regression_metrics(y,p,g),'rate_summary':{'min':float(r.min()),'median':float(np.median(r)),'max':float(r.max())},
  'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=y,prediction=p,groups=g);print(json.dumps({'selected':selected,'test':result['test']['pooled']},indent=2))
if __name__=='__main__':main()
