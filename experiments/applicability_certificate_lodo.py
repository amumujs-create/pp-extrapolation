#!/usr/bin/env python3
"""Retrospective cross-domain audit of a validation-only PP applicability certificate."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization
from pp_extrapolation.model import _affine_prediction
from distance_uncertainty_pp import prepare_battery

ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from run_affine_tail_external_three import prepare_hust,prepare_virkler
from pae_boundary_realdata import prepare_dataset
from femto_bearing_loader import FEATURE_COLS,load_femto_phm2012
SEEDS=(42,43,44,45,46)
def femto():
 df,g=load_femto_phm2012(root=Path('data/femto/raw'),file_stride=5)
 def part(units):
  d=df[df.unit.isin(units)];return {'x':d[FEATURE_COLS].to_numpy(np.float32),'y':d.RUL.to_numpy(np.float32),'groups':d.bearing.to_numpy()}
 tr,va=part(g['train']),part(g['val']);d=df[df.unit.isin(g['test'])].sort_values('cycle').groupby('unit',as_index=False).tail(1)
 te={'x':d[FEATURE_COLS].to_numpy(np.float32),'y':d.RUL.to_numpy(np.float32),'groups':d.bearing.to_numpy()};return tr,va,te
def audit(name,tr,va,te):
 affine=select_affine_initialization(tr,va);av=_affine_prediction(affine['initialization'],va['x'],affine['center'],affine['scale'],affine['target_scale']);at=_affine_prediction(affine['initialization'],te['x'],affine['center'],affine['scale'],affine['target_scale'])
 fits=[fit_pp(tr,va,seed=s,affine_selection=affine,max_epochs=300) for s in SEEDS];vp=np.asarray([predict(f,va['x']) for f in fits]);tp=np.asarray([predict(f,te['x']) for f in fits])
 affine_mse=float(np.mean((av-va['y'])**2));pp_seed_mse=float(np.mean((vp-va['y'][None,:])**2));gain=(affine_mse-pp_seed_mse)/max(affine_mse,1e-12)
 activity=float(np.mean(np.std(vp-av[None,:],axis=1))/max(np.std(va['y']),1e-12));disagreement=float(np.mean(np.std(vp,axis=0))/max(np.std(va['y']),1e-12))
 correlations=[abs(float(spearmanr(va['x'][:,j],va['y']).statistic)) for j in range(va['x'].shape[1])];max_corr=float(np.nanmax(correlations))
 checks={'relative_validation_gain_ge_1pct':gain>=.01,'residual_activity_ge_0.5pct':activity>=.005,'seed_disagreement_le_0.5':disagreement<=.5,'max_feature_rul_spearman_ge_0.2':max_corr>=.2}
 accepted=all(checks.values());ridge=regression_metrics(te['y'],at,te['groups'])['pooled'];pp=regression_metrics(te['y'],tp.mean(0),te['groups'])['pooled'];success=pp['r2']>ridge['r2'] and pp['r2']>0
 print(name,'accept',accepted,'gain',gain,'activity',activity,'corr',max_corr,'test',pp['r2'],flush=True)
 return {'validation_only':{'relative_pp_mse_gain':gain,'residual_activity':activity,'normalized_seed_disagreement':disagreement,'max_abs_feature_rul_spearman':max_corr,'checks':checks,'accepted':accepted},'test_audit':{'ridge':ridge,'pp_ensemble':pp,'success_definition':success},'selected_epochs':[f.selection['selected_epoch'] for f in fits]}
def main():
 torch.set_num_threads(2);ds={}
 for n,p in [('hust',prepare_hust()),('virkler',prepare_virkler())]:ds[n]=audit(n,*p[:3])
 for n in ('sunwoda','rwth','mich'):
  raw,_=prepare_dataset(n);ds[n]=audit(n,*prepare_battery(raw))
 ds['femto']=audit('femto',*femto())
 labels=[(d['validation_only']['accepted'],d['test_audit']['success_definition']) for d in ds.values()];payload={'experiment':'pp_applicability_certificate_v1','status':'retrospective development audit','fixed_checks':['validation PP gain >=1%','residual activity >=0.5% of validation target SD','seed disagreement <=0.5','max feature-RUL Spearman >=0.2'],'datasets':ds,'confusion':{'correct':sum(a==b for a,b in labels),'total':len(labels),'false_accept':sum(a and not b for a,b in labels),'false_reject':sum((not a) and b for a,b in labels)}}
 out=Path('results/applicability_certificate_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(payload['confusion'])
if __name__=='__main__':main()
