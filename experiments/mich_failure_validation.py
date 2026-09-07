#!/usr/bin/env python3
"""Recompute the recorded evidence for PP's MICH failure classification."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--pae-root',type=Path,default=Path(__file__).resolve().parents[2]/'ca-css-ncmapss');a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 base=json.load(open(root/'results/additional_real_batteries/results.json'))['datasets']['mich']
 cert=json.load(open(root/'results/applicability_certificate_v1/results.json'))['datasets']['mich']
 transport=json.load(open(root/'results/output_calibration_cross_dataset_v1/results.json'))['datasets']['mich']
 pae=json.load(open(a.pae_root/'results/pae_shared_battery_v1/results.json'))
 verify=json.load(open(a.pae_root/'results/pae_shared_battery_v1/VERIFICATION.json'))
 pp_r2=base['summary']['pp']['pooled_r2']['mean'];ridge_r2=base['summary']['ridge']['pooled_r2']['mean']
 gated=pae['summary']['boundary_gated_shared_nn']['datasets']['mich']
 result={
  'classification':'representation/shape failure for baseline PP; not an output-scale failure',
  'strict_hull':{'validation_outside_fraction':base['hull']['validation']['outside_fraction'],
   'test_outside_fraction':base['hull']['test']['outside_fraction'],
   'validation_distance_median':base['hull']['validation']['distance_median'],
   'test_distance_median':base['hull']['test']['distance_median']},
  'pp_collapse':{'relative_validation_gain':cert['validation_only']['relative_pp_mse_gain'],
   'normalized_residual_activity':cert['validation_only']['residual_activity'],
   'selected_epochs':cert['selected_epochs'],'pp_test_r2':pp_r2,'ridge_test_r2':ridge_r2,
   'absolute_pp_ridge_r2_difference':abs(pp_r2-ridge_r2)},
  'transport_rejection':{'seed_choices':[x['kind'] for x in transport['choices']],
   'validation_groups':transport['group_evidence']['groups'],'affine_group_wins':transport['group_evidence']['wins'],
   'mean_validation_mse_gain':transport['group_evidence']['mean_mse_gain'],
   'bootstrap_ci':transport['group_evidence']['bootstrap_ci'],'approved':transport['consensus_approved']},
  'alternative_structure_control':{'paper':'PAE, kept separate from PP',
   'boundary_gated_macro_unit_r2_mean':gated['macro_unit_r2']['mean'],
   'boundary_gated_macro_unit_r2_sd':gated['macro_unit_r2']['sd'],
   'boundary_gated_pooled_r2_mean':gated['pooled_r2']['mean'],
   'verification_all_checks_pass':verify['all_checks_pass']},
 }
 checks={
  'all_pp_checkpoints_epoch_zero':all(x==0 for x in result['pp_collapse']['selected_epochs']),
  'pp_equals_ridge_to_1e_minus_6':result['pp_collapse']['absolute_pp_ridge_r2_difference']<1e-6,
  'residual_activity_below_0_5pct':result['pp_collapse']['normalized_residual_activity']<.005,
  'all_transport_seeds_identity':all(x=='identity' for x in result['transport_rejection']['seed_choices']),
  'affine_bootstrap_upper_below_zero':result['transport_rejection']['bootstrap_ci'][1]<0,
  'test_farther_than_validation':result['strict_hull']['test_distance_median']>result['strict_hull']['validation_distance_median'],
  'alternative_boundary_model_positive':result['alternative_structure_control']['boundary_gated_pooled_r2_mean']>0,
 }
 result['checks']=checks;result['all_checks_pass']=all(checks.values())
 out=root/'results/mich_failure_validation_v1';out.mkdir(parents=True,exist_ok=True)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if not result['all_checks_pass']:raise SystemExit('MICH failure classification check failed')
if __name__=='__main__':main()
