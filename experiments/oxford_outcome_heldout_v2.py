#!/usr/bin/env python3
"""One-shot evaluation of frozen Oxford outcome-held-out protocol v2."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np,torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.benchmark_deps'),str(ROOT/'src'),str(ROOT/'experiments')]
from engression import engression
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from oxford_untouched import load_capacity,sha256,MAT
OUT=ROOT/'results/oxford_outcome_heldout_v2';SEEDS=range(42,47);PRESETS=('short','multiscale','moments');SEPS=(0.,.001,.01);ALPHAS=(.1,1.,10.,100.,1000.)
def feats(h,preset):
 out=[];inds=[]
 for i in range(7,len(h)):
  b=[h[i]];hs=(3,) if preset=='short' else (3,5,8)
  for k in hs:
   z=h[i-k+1:i+1];b += [float(z.mean()),float((z[-1]-z[0])/max(k-1,1))]
   if preset!='short':b += [float(z.std())]
  if preset=='moments':
   d=np.diff(h[i-7:i+1]);b += [float(d[-3:].mean()),float(d[-1]-d[0])]
  out.append(b);inds.append(i)
 return np.asarray(out,np.float32),np.asarray(inds)
def rows(cells,ids,preset,cut,side):
 X=[];Y=[];G=[]
 for i in ids:
  h=np.asarray(cells[i]/cells[i][0],float);x,ix=feats(h,preset);m=h[ix]>cut if side=='train' else h[ix]<cut;X.append(x[m]);Y.append((len(h)-1-ix[m]).astype(np.float32));G += [f'Cell{i}']*int(m.sum())
 return {'x':np.concatenate(X),'y':np.concatenate(Y),'groups':np.asarray(G)}
def r2pack(y,p,g):return regression_metrics(y,np.asarray(p),g)
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True);cells=load_capacity();pool=np.concatenate([cells[i][7:]/cells[i][0] for i in range(1,5)]);cut=float(np.quantile(pool,.5));data={p:(rows(cells,range(1,5),p,cut,'train'),rows(cells,(5,6),p,cut,'tail'),rows(cells,(7,8),p,cut,'tail')) for p in PRESETS}
 if cut!=0.8396755456924438 or len(data['short'][1]['y'])<8 or len(data['short'][2]['y'])<8:raise RuntimeError('frozen feasibility mismatch')
 # PP candidate fits; materialize test prediction only after validation selection.
 searches={};chosen=[];adaptive=[];fits={}
 for seed in SEEDS:
  rows_s=[]
  for preset in PRESETS:
   tr,va,te=data[preset];aff=select_affine_initialization(tr,va)
   for sep in SEPS:
    f=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=400,patience=80,separation_weight=sep,gate_weight=0.,width=32);key=(seed,preset,sep);fits[key]=f;rows_s.append({'preset':preset,'separation':sep,'validation_mse':f.selection['validation_mse']})
  best=min(rows_s,key=lambda z:z['validation_mse']);chosen.append(best);adaptive.append(predict_latent_regime(fits[(seed,best['preset'],best['separation'])],data[best['preset']][2]['x']));searches[str(seed)]=rows_s;print('PP',seed,best,flush=True)
 mean_cfg=[]
 for p in PRESETS:
  for s in SEPS:mean_cfg.append({'preset':p,'separation':s,'mean_validation_mse':float(np.mean([next(z['validation_mse'] for z in searches[str(seed)] if z['preset']==p and z['separation']==s) for seed in SEEDS]))})
 fixed=min(mean_cfg,key=lambda z:z['mean_validation_mse']);fixed_pred=[predict_latent_regime(fits[(seed,fixed['preset'],fixed['separation'])],data[fixed['preset']][2]['x']) for seed in SEEDS]
 y=data['short'][2]['y'];g=data['short'][2]['groups'];adaptive=np.asarray(adaptive);fixed_pred=np.asarray(fixed_pred)
 # Ridge searches over the same causal presets.
 rr=[]
 for p in PRESETS:
  tr,va,te=data[p]
  for a in ALPHAS:
   m=make_pipeline(StandardScaler(),Ridge(alpha=a)).fit(tr['x'],tr['y']);rr.append((float(np.mean((m.predict(va['x'])-va['y'])**2)),p,a,m))
 rb=min(rr,key=lambda z:z[0]);rp=np.clip(rb[3].predict(data[rb[1]][2]['x']),0,max(data[rb[1]][0]['y']))
 # Official Engression, frozen grid.
 tr,va,te=data['multiscale'];grid=[(h,lr,b,l,e) for h in (32,64) for lr in (.001,.005) for b in (.5,1.) for l,e in ((2,250),(3,500))];es=[]
 def ef(cfg,seed):
  torch.manual_seed(seed);h,lr,b,l,e=cfg;m=engression(torch.tensor(tr['x']),torch.tensor(tr['y'][:,None]),num_layer=l,hidden_dim=h,noise_dim=32,beta=b,lr=lr,num_epoches=e,batch_size=min(512,len(tr['y'])),device='cpu',standardize=True,verbose=False)
  with torch.no_grad():v=m.predict(torch.tensor(va['x']),target='mean',sample_size=50).squeeze().numpy();q=m.predict(torch.tensor(te['x']),target='mean',sample_size=100).squeeze().numpy()
  cap=max(tr['y']);return float(np.mean((np.clip(v,0,cap)-va['y'])**2)),np.clip(q,0,cap)
 for c in grid:es.append((ef(c,42)[0],c))
 ec=min(es,key=lambda z:z[0]);ep=np.asarray([ef(ec[1],s)[1] for s in SEEDS]);print('ENG',ec,flush=True)
 th=data['short'][0]['x'][:,0];vh=data['short'][1]['x'][:,0];qh=data['short'][2]['x'][:,0];sd=max(th.std(),1e-8);dist=np.maximum(th.min()-qh,qh-th.max())/sd
 res={'status':'one-shot scored after protocol commit 864c2b4','protocol_commit':'864c2b4','archive_sha256':sha256(MAT),'boundary':cut,'n':{'train':len(data['short'][0]['y']),'validation':len(data['short'][1]['y']),'test':len(y)},'hull':{'test_outside_fraction':float(np.mean(dist>0)),'median_distance_sd':float(np.median(dist)),'max_distance_sd':float(np.max(dist))},'pp':{'search':searches,'per_seed_selected':chosen,'adaptive_ensemble':r2pack(y,adaptive.mean(0),g),'adaptive_per_seed':[r2pack(y,p,g) for p in adaptive],'mean_validation_config':fixed,'fixed_ensemble':r2pack(y,fixed_pred.mean(0),g)},'ridge':{'selected_preset':rb[1],'alpha':rb[2],'validation_mse':rb[0],'metrics':r2pack(y,rp,g)},'engression':{'version':'0.1.9','selected':ec[1],'validation_mse':ec[0],'ensemble':r2pack(y,ep.mean(0),g),'per_seed':[r2pack(y,p,g) for p in ep]},'success':None}
 pp=res['pp']['fixed_ensemble']['pooled'];res['success']=bool(pp['r2']>0 and pp['rmse']<res['ridge']['metrics']['pooled']['rmse'] and pp['rmse']<res['engression']['ensemble']['pooled']['rmse']);(OUT/'results.json').write_text(json.dumps(res,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=y,groups=g,pp_adaptive=adaptive,pp_fixed=fixed_pred,engression=ep,ridge=rp);print('FINAL',res['success'],res['pp']['adaptive_ensemble']['pooled']['r2'],pp['r2'],res['ridge']['metrics']['pooled']['r2'],res['engression']['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
