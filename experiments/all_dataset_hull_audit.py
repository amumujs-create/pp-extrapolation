#!/usr/bin/env python3
"""Common extrapolation-severity audit on every PP benchmark split."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import audit_convex_hull_support
from extrapolation_competitors_all import datasets

def nn_ratio(train,source,seed=42):
 rng=np.random.default_rng(seed);x=np.asarray(train,float);q=np.asarray(source,float);mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);x=(x-mu)/sd;q=(q-mu)/sd
 if len(x)>5000:x=x[rng.choice(len(x),5000,replace=False)]
 nn=NearestNeighbors(n_neighbors=2).fit(x);base=np.median(nn.kneighbors(x,return_distance=True)[0][:,1]);d=NearestNeighbors(n_neighbors=1).fit(x).kneighbors(q,return_distance=True)[0][:,0]
 return {'median':float(np.median(d)),'p95':float(np.quantile(d,.95)),'max':float(d.max()),'relative_to_train_nn':float(np.median(d)/max(base,1e-8))}

def part(train,source):
 h=audit_convex_hull_support(train['x'][:,[0]],source['x'][:,[0]]).summary();h['full_feature_nn']=nn_ratio(train['x'],source['x']);h['target_outside_train_fraction']=float(np.mean((source['y']<np.min(train['y']))|(source['y']>np.max(train['y']))));return h

def one(parts):
 tr,va,te=parts;v=part(tr,va);t=part(tr,te);dv=v['distance_median'];dt=t['distance_median'];direction=float(np.sign(np.mean(va['x'][:,0])-np.mean(tr['x'][:,0]))*np.sign(np.mean(te['x'][:,0])-np.mean(tr['x'][:,0])))
 return {'n':{'train':len(tr['y']),'validation':len(va['y']),'test':len(te['y'])},'validation':v,'test':t,'test_to_validation_distance_ratio':None if dv<=1e-12 else float(dt/dv),'same_coordinate_ray':direction>=0}

def main():
 out={k:one(v) for k,v in datasets().items()}
 from run_affine_tail_external_nasa_health_v2 import prepare_folds
 folds,_=prepare_folds();out['nasa']={'folds':[{'test_cell':f['test_cell'],**one((f['train'],f['validation'],f['test']))} for f in folds]}
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_benchmark import rows
 s=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);names=list(FEATURE_COLS);out['ncmapss']=one((rows(s.train,names),rows(s.val,names),rows(s.test,names)))
 target=ROOT/'results/all_dataset_hull_audit_v1';target.mkdir(parents=True,exist_ok=True);(target/'results.json').write_text(json.dumps({'coordinate':'predeclared first PP extrapolation coordinate, standardized by train SD','datasets':out},indent=2)+'\n');print({k:(round(v.get('test',{}).get('outside_fraction',-1),3),round(v.get('test',{}).get('distance_median',-1),3)) for k,v in out.items()})
if __name__=='__main__':main()
