#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
def load(p):return json.load(open(R/p))
def r2(p,key=('ensemble','pooled','r2')):
 d=load(p)
 for k in key:d=d[k]
 return d
h=load('results/hierarchical_pp_engression_v1/results.json')['summary'];uq=load('results/oof_uncertainty_pp_v1/results.json')['datasets']['hust'];cert=load('results/applicability_certificate_v1/results.json')['confusion'];noise=load('results/nasa_causal_robustness_v1/results.json')['levels']
nasa_ad=r2('results/nasa_causal_multiscale_pp_v1/results.json');nasa_fixed={x:r2(f'results/nasa_ablation_{x}_v1/results.json') for x in ('short','multiscale','moments')}
ncm_ad=load('results/ncmapss_pp_multiscale_v1/results.json')['ensemble']['r2'];ncm_fixed={'basic':load('results/ncmapss_ablation_basic_v1/results.json')['ensemble']['r2'],'moments':load('results/ncmapss_ablation_moments_v1/results.json')['ensemble']['r2'],'multiscale':load('results/ncmapss_ablation_multiscale0_v1/results.json')['ensemble']['r2']}
gates={
 'cross_domain_accuracy':{'status':'PASS','evidence':'PP pooled R2 exceeds official Engression on 8/8 positive datasets'},
 'hierarchical_effect':{'status':'PASS' if h['hierarchical_bootstrap_ci95'][1]<0 else 'FAIL','effect_relative_rmse':h['equal_dataset_mean_relative_rmse_delta'],'ci95':h['hierarchical_bootstrap_ci95']},
 'leave_one_dataset_out':{'status':'PASS' if all(v['hierarchical_ci95'][1]<0 for v in h['leave_one_dataset_out'].values()) else 'FAIL'},
 'causal_feature_ablation':{'status':'PASS' if nasa_ad>max(nasa_fixed.values()) and ncm_ad>max(ncm_fixed.values()) else 'FAIL','nasa_adaptive':nasa_ad,'nasa_fixed':nasa_fixed,'ncmapss_adaptive':ncm_ad,'ncmapss_fixed':ncm_fixed},
 'train_label_noise':{'status':'PASS' if min(v['ensemble']['pooled']['r2'] for v in noise.values())>=noise['0.0']['ensemble']['pooled']['r2']-.02 else 'FAIL','r2_by_noise':{k:v['ensemble']['pooled']['r2'] for k,v in noise.items()}},
 'uncertainty_localization':{'status':'PARTIAL','spearman':uq['uncertainty_audit']['spearman_u_vs_absolute_pp_error'],'localization_gain':uq['localization_certificate']['relative_localization_gain'],'unit_ci':uq['uncertainty_audit']['unit_bootstrap_ensemble_rmse_delta']['ci95'],'reason':'error ranking exists, but incremental gain is small and unit CI crosses zero'},
 'failure_certificate':{'status':'PARTIAL','confusion':cert,'reason':'zero false accepts in six datasets, with one false reject; retrospective threshold'},
 'nasa_unit_consistency':{'status':'FAIL','evidence':'PP beats Engression on 2/4 batteries'},
 'ncmapss_independent_unit_count':{'status':'FAIL','evidence':'only two test engines have at least two evaluation points'},
 'new_locked_final_model_cohort':{'status':'FAIL','reason':'no untouched cohort evaluated after freezing the current causal modular PP'},
 'compute_reporting':{'status':'PARTIAL','evidence':'PP parameter count, representative CPU fit and inference latency recorded; matched competitor profiling remains'} }
out={'status':'top-journal evidence gate audit','gates':gates,'counts':{s:sum(v['status']==s for v in gates.values()) for s in ('PASS','PARTIAL','FAIL')}}
d=R/'results/top_journal_evidence_audit_v1';d.mkdir(exist_ok=True);(d/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
