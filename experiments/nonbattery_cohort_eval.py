"""Frozen non-battery threshold-RUL comparison. Input: audited series JSON.
Each unit: time and health arrays (initial-reference normalized by extractor).
No synthetic interpolation rows; interpolate event time only.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,support_distance
from pp_extrapolation.model import _affine_prediction
from plain_mlp_ablation import fit_plain,predict_plain
from cross_domain_mechanism_study import paired


def event_series(e):
 t=np.asarray(e['time'],float);h=np.asarray(e['health'],float)
 assert len(t)==len(h) and np.all(np.diff(t)>0) and np.isfinite(h).all()
 hit=np.flatnonzero(h<=.8)
 if not len(hit) or hit[0]==0:return None
 j=hit[0];end=t[j-1]+(h[j-1]-.8)/(h[j-1]-h[j])*(t[j]-t[j-1])
 return t[:j],h[:j],float(end),[float(t[j-1]),float(t[j])]

def rows(series,ids,train,scale,mode="strict"):

 x=[];y=[];g=[]
 for uid in ids:
  e=event_series(series[uid])
  if e is None:continue
  t,h,end,_=e;t=t-t[0];elapsed=t/scale
  for i in range(len(t)):
   if mode=='strict' and not (h[i]>.9 if train else .8<h[i]<=.9):continue
   slopes=[(h[i]-h[max(0,i-w)])/(elapsed[i]-elapsed[max(0,i-w)]+1e-8) for w in (1,2,3)]
   hist=h[max(0,i-3):i+1]
   x.append([h[i],elapsed[i],*slopes,hist.mean(),hist.std(),h[i]-h[0],min(h[:i+1]),min(i,3)/3.,float(i==0)])
   y.append(end-(t[i]+np.asarray(series[uid]['time'])[0]));g.append(uid)
 return {'x':np.asarray(x,np.float32).reshape(-1,11),'y':np.asarray(y,np.float32),'groups':np.asarray(g)}

def main(path,mode="strict"):
 torch.set_num_threads(2);p=Path(path);series=json.loads(p.read_text());keys=sorted(series);n=len(keys);a=int(.6*n);b=a+int(.2*n)
 ids={'train':keys[:a],'validation':keys[a:b],'test':keys[b:]};out=ROOT/'results'/f'{p.stem}_{mode}_locked_v1';out.mkdir(parents=True,exist_ok=True)
 audit={u:({'event_time':e[2],'event_bracket':e[3],'n_pre_event':len(e[0])} if (e:=event_series(series[u])) else {'censored':True}) for u in keys}
 scale=max(float(np.max(series[u]['time'])-np.min(series[u]['time'])) for u in ids['train'])
 tr,va,te=[rows(series,ids[k],k=='train',scale,mode) for k in ('train','validation','test')]
 counts={k:{u:int(np.sum(r['groups']==u)) for u in ids[k]} for k,r in zip(ids,(tr,va,te))}
 result={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'split_ids':ids,'audit':audit,'row_counts':counts,'protocol':'NONBATTERY_EXTERNAL_PROTOCOL.md','mode':mode}
 good=[sum(v>=3 for v in counts[k].values()) for k in ids]
 if any(v<need for v,need in zip(good,(2,1,2))) or any(0<v<3 for d in counts.values() for v in d.values()):
  result['status']='not evaluable under frozen adequacy criteria';(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
 aff=select_affine_initialization(tr,va);models={};preds={};valpred={}
 configs={'PP':{},**{f'safe_{lam}':dict(direct_residual_mixture=True,fixed_affine_trust=lam,residual_seed_replay=True,residual_zero_init=False) for lam in (0.,.02,.05,.1,.2)}}
 # Test labels are not read for scoring until all fitting/selection ends.
 for name,cfg in {'MLP':None,**configs}.items():
  vv=[];tt=[]
  for seed in (42,43,44,45,46):
   fit=fit_plain(tr,va,seed=seed) if cfg is None else fit_pp(tr,va,seed=seed,affine_selection=aff,**cfg)
   fn=predict_plain if cfg is None else predict
   vv.append(fn(fit,va['x']));tt.append(fn(fit,te['x']))
  valpred[name]=np.asarray(vv);preds[name]=np.asarray(tt)
  print('finished',name,flush=True)
 choice=min(configs.keys()-{'PP'},key=lambda k:np.mean((valpred[k].mean(0)-va['y'])**2))
 result['validation_selected_safe_route']=choice
 ridge=_affine_prediction(aff['initialization'],te['x'],aff['center'],aff['scale'],aff['target_scale']);preds['ridge']=ridge[None,:]
 for name,pp in preds.items():
  models[name]={'ensemble':regression_metrics(te['y'],pp.mean(0),te['groups']),'seed_pooled_r2':[regression_metrics(te['y'],q,te['groups'])['pooled']['r2'] for q in pp]}
 d,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
 result.update(status='completed pilot',models=models,health_hull={'outside_fraction':float(np.mean(d>1e-8)),'median_distance':float(np.median(d))},paired_pp_vs_mlp=paired(te['y'],te['groups'],preds['MLP'].mean(0),preds['PP'].mean(0),np.random.default_rng(20260908)))
 (out/'results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'predictions.npz',y=te['y'],groups=te['groups'],distance=d,**preds);print(json.dumps(result,indent=2))
if __name__=='__main__':main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else "strict")
