import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from extended_nn_benchmark import data,OUT
from pp_extrapolation import regression_metrics
tr,va,te=data();cap=max(tr['y'].max(),1);OUT.mkdir(exist_ok=True)
with threadpool_limits(limits=1):
 for name in ['ridge','spline','boosting']:
  best=None;rows=[]
  for a in [0.01,1.,100.]:
   for b in [3,5,9]:
    if name=='ridge':m=make_pipeline(StandardScaler(),Ridge(alpha=a*b))
    elif name=='spline':m=make_pipeline(StandardScaler(),SplineTransformer(n_knots=b,degree=3,extrapolation='linear'),Ridge(alpha=a))
    else:m=HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=b*3,l2_regularization=a,early_stopping=False,random_state=42)
    m.fit(tr['x'],tr['y']);score=float(np.mean((np.clip(m.predict(va['x']),0,cap)-va['y'])**2));rows.append({'a':a,'b':b,'validation_mse':score})
    if best is None or score<best[0]:best=(score,m,a,b)
  raw=best[1].predict(te['x']);p=np.clip(raw,0,cap);(OUT/f'{name}.json').write_text(json.dumps({'search':rows,'selected':[best[2],best[3]],'ensemble':regression_metrics(te['y'],p,te['groups'])},indent=2));np.savez_compressed(OUT/f'{name}.npz',raw=raw,prediction=p,y=te['y'],groups=te['groups'])
np.savez_compressed(OUT/'pfn_input.npz',train_x=tr['x'],train_y=tr['y'],test_x=te['x'],test_y=te['y'],groups=te['groups'])
