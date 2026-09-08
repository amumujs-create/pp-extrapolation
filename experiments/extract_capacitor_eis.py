"""NASA ES10: median 100-Hz series capacitance across within-visit repeats.
Technical EIS repeats are not independent units. Duplicate visit ages are pooled.
"""
import json
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/nonbattery_external'
f=h5py.File(D/'ES10.mat');table=f['ES10/EIS_Data/EIS_Reference_Table'];times=[]
for r in table[3]:
 v=f[r];times.append(float(''.join(map(chr,v[()].ravel()))))
series={};audit={}
for uid in sorted(k for k in f['ES10/EIS_Data'] if k.startswith('ES10C')):
 m=f[f'ES10/EIS_Data/{uid}/EIS_Measurement'];visits=[]
 for j,t in enumerate(times):
  ds=f[m['Data'][j,0]];caps=[]
  if ds.dtype==object:
   for ref in ds[()].ravel():
    if not ref:continue
    arr=f[ref][()]
    if arr.ndim!=2 or arr.shape[0]<9:continue
    freq=arr[0];cs=arr[8];valid=np.isfinite(freq)&np.isfinite(cs)&(freq>0)&(cs>0)
    if valid.any():
     idx=np.flatnonzero(valid)[np.argmin(abs(freq[valid]-100))]
     if abs(freq[idx]-100)<5:caps.append(float(cs[idx]))
  if caps:visits.append((t,float(np.median(caps))))
 if not visits:
  audit[uid]={'reason':'no usable 100Hz EIS'};continue
 ts=sorted(set(t for t,c in visits));cs=[float(np.median([c for t,c in visits if t==u])) for u in ts]
 series[uid]={'time':ts,'health':(np.asarray(cs)/cs[0]).tolist(),'capacitance_uF':cs}
 audit[uid]={'n_visits':len(ts),'initial_C':cs[0]}
(D/'capacitor_es10.json').write_text(json.dumps(series,indent=2));(D/'capacitor_extraction_audit.json').write_text(json.dumps(audit,indent=2));print(audit)
