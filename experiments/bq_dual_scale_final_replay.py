#!/usr/bin/env python3
"""Frozen final replay and contraction audit for adaptive dual-scale PP."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_affine,predict_boundary_quotient
OUT=ROOT/'results/bq_dual_scale_final_replay_v1';SEEDS=(42,43,44,45,46)
CFG=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.,broad_residual_bound=6.,local_saturation_weight=.4,support_gate_feature=2,support_gate_threshold=.5,support_gate_temperature=.25,support_adaptive_saturation=True)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};preds=[];affines=[];runs=[]
 for seed in SEEDS:
  sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**CFG);ep=max(sel.selection['selected_epoch'],1);fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=ep,patience=10000,restore_best=False,**CFG);p=predict_boundary_quotient(fit,rows['source']);a=predict_boundary_affine(fit,rows['source']);preds.append(p);affines.append(a);runs.append({'seed':seed,'selected_epoch':ep,'validation_mse':sel.selection['validation_dataset_macro_mse'],'metrics':score_by_dataset(p,rows['source'],scales)});print(seed,flush=True)
 P=np.asarray(preds);A=np.asarray(affines);en=score_by_dataset(P.mean(0),rows['source'],scales);envelope=rows['source']['margin'][None,:]*CFG['broad_residual_bound'];excess=np.abs(P-A)-envelope;result={'status':'frozen replay of retrospectively selected final PP','config':CFG,'runs':runs,'ensemble':en,'contraction_audit':{'predictions_checked':int(P.size),'violations':int(np.count_nonzero(excess>1e-6)),'maximum_excess':float(excess.max()),'bound':'abs(yhat-y_affine) <= margin * 6'},'data_audits':audits,'runtime_seconds':time.perf_counter()-start};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=P,affine=A,y=rows['source']['y'],margin=rows['source']['margin'],units=rows['source']['units'],dataset=rows['source']['dataset']);print({d:round(en[d]['pooled_r2'],3) for d in DATASETS},result['contraction_audit'],flush=True)
if __name__=='__main__':main()
