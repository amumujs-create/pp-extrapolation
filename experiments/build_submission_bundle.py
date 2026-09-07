#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,platform,subprocess,sys
import numpy,scipy,sklearn,torch
R=Path(__file__).resolve().parents[1]
FILES=['results/engression_all_positive_v2/results.json','results/all_dataset_hull_audit_v1/results.json','results/hierarchical_pp_engression_v1/results.json','results/nasa_causal_multiscale_pp_v1/results.json','results/ncmapss_pp_multiscale_v1/results.json','results/nasa_causal_robustness_v1/results.json','results/top_journal_evidence_audit_v1/results.json','results/final_pp_compute_profile_v1/results.json','results/oxford_outcome_heldout_v2/results.json','results/oxford_compute_profile_v1/results.json']
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip())
 manifest={'bundle':'submission_bundle_v1','code_commit':commit,'dirty_at_manifest_build':dirty,'python':sys.version,'platform':platform.platform(),'packages':{'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},'artifacts':{p:{'sha256':sha(p),'bytes':(R/p).stat().st_size} for p in FILES},'primary_results':{'eight_domain_pp_vs_engression':'8/8 PP wins','hierarchical_relative_rmse_reduction':0.24047051762355423,'hierarchical_ci95_reduction':[0.055121614165963756,0.408613608486634],'nasa_pp_r2':0.5837508453043845,'ncmapss_pp_r2':0.9372705221176147,'oxford_pp_r2':-0.11934027429040661,'oxford_success':False},'commands':['PYTHONPATH=src:experiments:.benchmark_deps pytest -q','PYTHONPATH=src:experiments:.benchmark_deps:../ca-css-ncmapss python experiments/hierarchical_pp_engression_analysis.py','PYTHONPATH=src:experiments:.benchmark_deps python experiments/top_journal_evidence_audit.py','MPLCONFIGDIR=/tmp/pp-mpl PYTHONPATH=.benchmark_deps:src:experiments python experiments/oxford_outcome_heldout_v2.py'],'note':'Raw data and large predictions are ignored by git; artifact hashes bind the local result files. Oxford is a negative outcome-held-out external evaluation.'}
 out=R/'reproducibility/submission_bundle_v1';out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
