#!/usr/bin/env python3
"""Frozen group-LOO output calibration audit across PP datasets."""
from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
import numpy as np,torch
from pp_extrapolation import (LatentRegimeFit,LatentRegimePPNet,OutputCalibrator,
 approve_dual_evidence,fit_pp,group_loo_affine_evidence,predict,
 regression_metrics,select_affine_initialization,select_group_loo_calibrator,
 approve_seed_consensus)
SEEDS=(42,43,44,45,46)

def summarize(y,g,matrix):
 ms=[regression_metrics(y,p,g) for p in matrix];r=[m['pooled']['r2'] for m in ms]
 return {'mean_r2':statistics.mean(r),'sample_sd_r2':statistics.stdev(r),
         'ensemble':regression_metrics(y,np.mean(matrix,0),g),'per_seed':ms}

def calibrate(validation,test,validation_predictions,test_predictions,cap):
 calibrated=[];choices=[]
 for seed,vp,tp in zip(SEEDS,validation_predictions,test_predictions):
  if len(np.unique(validation['groups']))<3:
   from pp_extrapolation.output_calibration import OutputCalibrator
   cal=OutputCalibrator('identity');scores={'identity':float(np.mean((vp-validation['y'])**2))}
  else:cal,scores=select_group_loo_calibrator(validation['y'],vp,validation['groups'],upper=cap)
  calibrated.append(cal.predict(tp,upper=cap));choices.append({'seed':seed,'kind':cal.kind,'slope':cal.slope,'intercept':cal.intercept,'loo_mse':scores})
 return np.asarray(calibrated),choices

def base_dataset(parts):
 tr,va,te=parts;aff=select_affine_initialization(tr,va);vp=[];tp=[]
 for seed in SEEDS:
  fit=fit_pp(tr,va,seed=seed,affine_selection=aff,max_epochs=300);vp.append(predict(fit,va['x']));tp.append(predict(fit,te['x']))
 cal,choices=calibrate(va,te,np.asarray(vp),np.asarray(tp),float(aff['target_scale']))
 evidence=group_loo_affine_evidence(va['y'],np.mean(vp,axis=0),va['groups'],upper=float(aff['target_scale']))
 approved=approve_dual_evidence([OutputCalibrator(row['kind']) for row in choices],evidence)
 return {'choices':choices,'group_evidence':evidence,'consensus_approved':approved,'base':summarize(te['y'],te['groups'],np.asarray(tp)),'calibrated':summarize(te['y'],te['groups'],cal),'final':summarize(te['y'],te['groups'],cal if approved else np.asarray(tp))}

def ncmapss(h5,legacy):
 sys.path.insert(0,str(legacy));from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_css import eval_all_bands_hard
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_benchmark import pp_features,rows
 split=make_tra_hard_split(h5,max_windows_per_unit=1500,random_seed=42);names=list(FEATURE_COLS);va=rows(split.val,names);vx=va['x'];all_x=pp_features(split.all_windows,names);vp=[];tp=[]
 root=Path('results/ncmapss_pp_benchmark_v1')
 for seed in SEEDS:
  state=torch.load(root/f'seed{seed}.pt',map_location='cpu',weights_only=False);meta=json.load(open(root/f'seed{seed}.json'))['selected'];model=LatentRegimePPNet(len(state['center']),meta['direction'],meta['knot'],width=24);model.load_state_dict(state['model_state']);fit=LatentRegimeFit(model,state['center'],state['scale'],state['target_scale'],meta);vp.append(predict_latent(fit,vx));tp.append(predict_latent(fit,all_x))
 cal,choices=calibrate(va,None,np.asarray(vp),np.asarray(tp),float(fit.target_scale))
 def sm(matrix):
  vals=[eval_all_bands_hard(split,p)['hard_extrap']['overall'] for p in matrix];r=[v['r2'] for v in vals]
  return {'mean_r2':statistics.mean(r),'sample_sd_r2':statistics.stdev(r),'ensemble':eval_all_bands_hard(split,np.mean(matrix,0))['hard_extrap']['overall']}
 evidence=group_loo_affine_evidence(va['y'],np.mean(vp,axis=0),va['groups'],upper=float(fit.target_scale))
 approved=approve_dual_evidence([OutputCalibrator(row['kind']) for row in choices],evidence)
 return {'choices':choices,'group_evidence':evidence,'consensus_approved':approved,'base':sm(np.asarray(tp)),'calibrated':sm(cal),'final':sm(cal if approved else np.asarray(tp))}

def predict_latent(fit,x):
 from pp_extrapolation import predict_latent_regime
 return predict_latent_regime(fit,x)

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-root',type=Path,default=Path(__file__).resolve().parents[2]/'ca-css-ncmapss');p.add_argument('--h5',type=Path,default=Path('data/N-CMAPSS_DS02-006.h5'));a=p.parse_args();legacy=a.legacy_root.resolve();sys.path.insert(0,str(legacy));torch.set_num_threads(2)
 from pae_boundary_realdata import prepare_dataset
 from run_affine_tail_external_three import prepare_hust,prepare_virkler
 from distance_uncertainty_pp import prepare_battery
 datasets={}
 for name,parts in [('hust',prepare_hust()[:3]),('virkler',prepare_virkler()[:3])]:datasets[name]=base_dataset(parts);print(name,datasets[name]['base']['ensemble']['pooled']['r2'],datasets[name]['calibrated']['ensemble']['pooled']['r2'],flush=True)
 for name in ('sunwoda','rwth','mich'):
  raw,_=prepare_dataset(name);datasets[name]=base_dataset(prepare_battery(raw));print(name,datasets[name]['base']['ensemble']['pooled']['r2'],datasets[name]['calibrated']['ensemble']['pooled']['r2'],flush=True)
 if a.h5.exists():datasets['ncmapss']=ncmapss(a.h5.resolve(),legacy);print('ncmapss',datasets['ncmapss']['base']['ensemble']['r2'],datasets['ncmapss']['calibrated']['ensemble']['r2'],flush=True)
 out=Path('results/output_calibration_cross_dataset_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps({'status':'post-hoc cross-dataset audit; one fixed calibration rule','datasets':datasets},indent=2)+'\n')
if __name__=='__main__':main()
