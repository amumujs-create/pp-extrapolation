"""Cross-dataset transfer of the fixed extended PP/FT benchmark protocol."""
import sys,json
from pathlib import Path
import numpy as np
import extended_nn_benchmark as b
from regime_spline_deep_future import splits

def main():
 root=Path('results/cross_dataset_pp_ft_v1');root.mkdir(parents=True,exist_ok=True)
 for name,parts in splits().items():
  for p in parts:p['ids']=np.array([f'{g}:row{i}' for i,g in enumerate(p['groups'])])
  b.OUT=root/name;b.OUT.mkdir(parents=True,exist_ok=True)
  (b.OUT/'split.json').write_text(json.dumps({'counts':[len(p['y']) for p in parts],'units':[len(np.unique(p['groups'])) for p in parts],'coordinate_ranges':[[float(p['x'][:,0].min()),float(p['x'][:,0].max())] for p in parts],'status':'previously inspected development datasets'},indent=2))
  b.data=lambda parts=parts:parts
  sys.argv=['extended_nn_benchmark','--models','pp,pp_joint,ft_transformer'];b.main()
 # MATR uses already completed identical-protocol benchmark; never relabel historical 0.257.
 rows=[]
 for name in ['hust','virkler','matr2019']:
  folder=Path('results/extended_nn_benchmark_v1') if name=='matr2019' else root/name
  for kind in ['pp','pp_joint','ft_transformer']:
   r=json.load(open(folder/f'{kind}.json'));rows.append({'dataset':name,'model':kind,'ensemble_r2':r['ensemble']['pooled']['r2'],'mean_r2':r['mean_r2'],'sd_r2':r['sd_r2'],'rmse':r['ensemble']['pooled']['rmse'],'source':str(folder/f'{kind}.json')})
 (root/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
