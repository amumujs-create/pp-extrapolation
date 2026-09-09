"""Freeze the PP-X FEMTO route without reading target-test labels for routing."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pp_extrapolation import PriorEvidence,select_ppx_route
OUT=ROOT/'results/femto_ppx_final_v14'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Counts and boundary availability come from the Learning-set contract.
    # The rejected mode model is not needed to make this structural decision.
    evidence=PriorEvidence(known_boundary=False,complete_groups=6,
      minimum_complete_groups_per_regime=2,oof_prior_regret=None,oof_mode_stability=None)
    decision=select_ppx_route(evidence,min_groups=5,min_per_regime=3)
    manifest={'status':'retrospective final development route; not untouched confirmation',
      'model':'PP-X: transferability-gated prior portfolio with neural safety executor',
      'evaluation_scope':'inductive extrapolation; frozen source-trained model',
      'test_time_adaptation':False,
      'test_batch_statistics_used':False,
      'test_prefix_training_or_self_supervision':False,
      'test_unit_information_sharing':False,
      'decision':decision.__dict__,'thresholds':{'min_complete_groups':5,'min_per_regime':3,
        'max_oof_prior_regret':0.,'min_mode_stability':.6},
      'route_inputs':'Learning metadata and source evidence only; target-test y is not an input',
      'candidate_prediction_artifact':'results/femto_waveform_pp_v8/direct.npz',
      'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'route_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if decision.route!='neural_safety':raise RuntimeError('FEMTO prior should not be approved by frozen evidence')
    z=np.load(ROOT/'results/femto_waveform_pp_v8/direct.npz');P=np.asarray(z['predictions']);y=np.asarray(z['y'])
    result={'model':'PP-X','route':decision.route,'prior_weight':decision.prior_weight,
      'pooled_r2':float(r2_score(y,P.mean(0))),'rmse':float(np.sqrt(np.mean((y-P.mean(0))**2))),
      'seed_r2':[float(r2_score(y,p)) for p in P],
      'interpretation':'framework-level FEMTO coverage through prior abstention; not evidence that a physical prior improves FEMTO'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'predictions.npz',prediction=P,y=y,groups=z['groups'])
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
