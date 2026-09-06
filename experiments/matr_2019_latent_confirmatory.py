#!/usr/bin/env python3
"""One-shot MATR 2019 latent-transition PP confirmatory evaluation."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import h5py,numpy as np,torch
from pp_extrapolation import (fit_latent_regime_pp,fit_pp,fit_regime_spline_pp,
 jacobian_diagnostics,predict,predict_latent_regime,predict_regime_spline,
 regression_metrics,select_affine_initialization,support_distance)
from pp_extrapolation.model import _affine_prediction
from plain_mlp_ablation import fit_plain,predict_plain
ROOT=Path(__file__).resolve().parents[1];MAT=ROOT/'data/matr/2019-01-24_batchdata_updated_struct_errorcorrect.mat';OUT=ROOT/'results/matr_2019_latent_confirmatory'
SEEDS=(42,43,44,45,46);JW=(0.,.01,.1,1.);SW=(0.,.001,.01);GW=(0.,.001,.01)
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_cells():
 cells=[];excluded=[]
 with h5py.File(MAT,'r') as f:
  refs=f['batch/summary'];n=refs.shape[0]
  for i in range(n):
   try:
    s=f[refs[i,0]];q=np.asarray(s['QDischarge']).reshape(-1).astype(np.float32);cycle=np.asarray(s['cycle']).reshape(-1).astype(np.float32)
    ok=np.isfinite(q)&np.isfinite(cycle);q,cycle=q[ok],cycle[ok]
    if len(q)<24:excluded.append({'index':i,'reason':'fewer than 24 finite observations'});continue
    cells.append({'index':i,'q':q,'cycle':cycle})
   except Exception as e:excluded.append({'index':i,'reason':type(e).__name__})
 return cells,excluded,n
def make_rows(cells,boundary,train=False,targets=True):
 xs=[];ys=[];groups=[];coordinate=[]
 for c in cells:
  q,cy=c['q'],c['cycle'];rate=np.r_[0.,np.maximum(q[:-1]-q[1:],0.)]
  for end in range(7,len(q)):
   if (q[end]>boundary) != train:continue
   h=q[end-7:end+1];r=rate[end-7:end+1];xs.append([h[-1],r[-1],h.mean(),r.mean(),h[-1]-h[0],r[-1]-r[0]]);coordinate.append(h[-1]);groups.append(f'c{c["index"]}')
   if targets:ys.append(float(cy[-1]-cy[end]))
 out={'x':np.asarray(xs,dtype=np.float32),'groups':np.asarray(groups),'coordinate':np.asarray(coordinate,dtype=np.float32)}
 if targets:out['y']=np.asarray(ys,dtype=np.float32)
 return out
def summary(rows,pred):return regression_metrics(rows['y'],pred,rows['groups'])
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);cells,excluded,nraw=load_cells()
 if len(cells)<12:raise RuntimeError('eligibility failed: fewer than 12 cells')
 nt=int(.6*len(cells));nv=int(.2*len(cells));tc,vc,sc=cells[:nt],cells[nt:nt+nv],cells[nt+nv:]
 endpoints=np.concatenate([c['q'][7:] for c in tc]);cut=float(np.quantile(endpoints,.25));train0=make_rows(tc,cut,train=True);boundary=float(train0['coordinate'].min());train=make_rows(tc,boundary,train=True);val=make_rows(vc,boundary);source=make_rows(sc,boundary,targets=False)
 vd,va=support_distance(train['coordinate'][:,None],val['coordinate'][:,None]);sd,sa=support_distance(train['coordinate'][:,None],source['coordinate'][:,None])
 audit={'protocol_commit':'ab103c8','raw_cells':nraw,'eligible_cells':len(cells),'excluded':excluded,'split_indices':{'train':[c['index'] for c in tc],'validation':[c['index'] for c in vc],'test':[c['index'] for c in sc]},'cutoff_quantile_value':cut,'actual_boundary':boundary,'counts':{'train':len(train['x']),'validation':len(val['x']),'source':len(source['x'])},'hull':{'validation':va.summary(),'source':sa.summary()}}
 (OUT/'eligibility_pretest.json').write_text(json.dumps(audit,indent=2)+'\n');print('AUDIT',audit,flush=True)
 if len(val['x'])<20 or len(source['x'])<20 or va.outside_fraction<1 or sa.outside_fraction<1:raise RuntimeError('locked feasibility conditions failed')
 affine=select_affine_initialization(train,val);pred={'pp':[],'plain':[],'jacobian':[],'latent':[]};runs={'pp':[],'plain':[],'jacobian':[],'latent':[]}
 for seed in SEEDS:
  pp=fit_pp(train,val,seed=seed,affine_selection=affine,max_epochs=300);pred['pp'].append(predict(pp,source['x']));runs['pp'].append(pp.selection)
  plain=fit_plain(train,val,seed=seed,max_epochs=300);pred['plain'].append(predict_plain(plain,source['x']))
  jc=[]
  for w in JW:
   f=fit_regime_spline_pp(train,val,seed=seed,affine_selection=affine,max_epochs=300,jacobian_weight=w,jacobian_ray_multiplier=1.);jc.append((f.selection['validation_mse'],w,f))
  _,w,j=min(jc,key=lambda z:(z[0],z[1]));jp=predict_regime_spline(j,source['x']);pred['jacobian'].append(jp);runs['jacobian'].append({'weight':w,'selection':j.selection,'source_jacobian':jacobian_diagnostics(j,source['x'])})
  lc=[]
  for sw in SW:
   for gw in GW:
    f=fit_latent_regime_pp(train,val,seed=seed,affine_selection=affine,separation_weight=sw,gate_weight=gw);lc.append((f.selection['validation_mse'],sw,gw,f))
  _,sw,gw,l=min(lc,key=lambda z:(z[0],z[1],z[2]));lp,g=predict_latent_regime(l,source['x'],return_gate=True);pred['latent'].append(lp);runs['latent'].append({'separation':sw,'gate_weight':gw,'selection':l.selection,'source_gate':{'mean':float(g.mean()),'sd':float(g.std()),'min':float(g.min()),'max':float(g.max())}});print('FIT',seed,w,sw,gw,flush=True)
 # Only here are untouched targets materialized.
 test=make_rows(sc,boundary,targets=True);ridge=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale']);result={'pretest':audit,'archive_sha256':sha256(MAT),'ridge':summary(test,ridge),'runs':runs}
 for name,values in pred.items():
  a=np.asarray(values);metrics=[summary(test,p) for p in a];rs=np.asarray([m['pooled']['r2'] for m in metrics]);result[name]={'single_seed_r2_mean':float(rs.mean()),'single_seed_r2_sd':float(rs.std()),'ensemble':summary(test,a.mean(0)),'seeds':metrics}
 lat=result['latent'];result['confirmatory_criterion']={'ensemble_beats_plain':lat['ensemble']['pooled']['r2']>result['plain']['ensemble']['pooled']['r2'],'ensemble_beats_pp':lat['ensemble']['pooled']['r2']>result['pp']['ensemble']['pooled']['r2'],'mean_beats_plain':lat['single_seed_r2_mean']>result['plain']['single_seed_r2_mean'],'mean_beats_pp':lat['single_seed_r2_mean']>result['pp']['single_seed_r2_mean'],'sd_no_greater_than_plain':lat['single_seed_r2_sd']<=result['plain']['single_seed_r2_sd']};result['confirmatory_success']=all(result['confirmatory_criterion'].values())
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['confirmatory_success'],{k:result[k]['ensemble']['pooled']['r2'] for k in pred},flush=True)
if __name__=='__main__':main()
