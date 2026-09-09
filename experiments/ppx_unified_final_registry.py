"""Materialize the development-final PP-X route registry.

Scores are frozen references to existing result artifacts. This script does not
retrain or silently reinterpret them as one matched experimental protocol.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ppx_unified_final_v1'
ROUTES=[
 ('HUST',.958,'regime_transport_pp','results/hust_regime_transport_pp_v1/results.json'),
 ('Virkler',.888,'support_gated_pp','results/support_gated_cross_domain_v1/results.json'),
 ('NASA battery',.584,'causal_multiscale_latent_pp','results/nasa_causal_multiscale_pp_v1/results.json'),
 ('Sunwoda',.939,'validation_approved_bounded_boundary_pp','results/bq_pp_matched_controls_v1/results.json'),
 ('RWTH',.878,'validation_approved_bounded_boundary_pp','results/bq_pp_matched_controls_v1/results.json'),
 ('MATR2019',.466,'validation_calibrated_latent_pp','results/matr_pp_validation_calibration_v1/results.json'),
 ('MATR batch 2',.862,'regime_transport_pp','results/matr_batch2_pp_five_seed_replay/results.json'),
 ('N-CMAPSS',.937,'causal_multiscale_latent_pp','results/ncmapss_pp_multiscale_v1/results.json'),
 ('MICH',.751,'support_adaptive_dual_scale_boundary_pp','results/bq_dual_scale_final_replay_v1/results.json'),
 ('XJTU',.257,'retrospective_reflected_scale_temporal_pp','results/xjtu_reflected_scale_pp_v3/results.json'),
 ('FEMTO',.07488203048706055,'transferability_gate_to_neural_safety','results/femto_ppx_final_v14/results.json'),
 ('NASA milling',.341,'inspection_calibrated_boundary_quotient','results/milling_boundary_quotient_route_v1/results.json'),
]
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for name,score,route,path in ROUTES:
  resolved=ROOT/path
  if not resolved.exists():raise FileNotFoundError(resolved)
  rows.append({'dataset':name,'development_pooled_r2':score,'route':route,'artifact':path})
 result={'model':'PP-X','definition':'prior portfolio + transferability gate + bounded neural correction/direct safety executor',
  'scope':'inductive extrapolation only; no TTA, transductive fitting, or test-batch statistics',
  'status':'retrospective development registry; heterogeneous dataset-specific protocols; not a new untouched confirmation',
  'positive_r2_routes':sum(r['development_pooled_r2']>0 for r in rows),'total_routes':len(rows),'routes':rows,
  'femto_caveat':'positive ensemble R2 comes from prior abstention; all five individual waveform-NN seeds have negative R2'}
 (OUT/'registry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
