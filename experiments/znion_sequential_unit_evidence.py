#!/usr/bin/env python3
"""Unit-level paired evidence across unique sequential Zn-ion cohorts."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/znion_sequential_unit_evidence'

def main():
 improved=json.loads((ROOT/'results/znion_rbf_regime_pp_alpha_development/results.json').read_text())['replay'];old2=json.loads((ROOT/'results/znion_rbf_regime_confirmation_v2/results.json').read_text());old3=json.loads((ROOT/'results/znion_rbf_regime_confirmation_v3/results.json').read_text());rows=[]
 for cohort,pp,plain in [('v2',improved['confirmation_v2']['metrics']['per_unit'],old2['plain_mlp']['per_unit']),('v3',improved['confirmation_v3']['metrics']['per_unit'],old3['plain_mlp']['per_unit'])]:
  for unit in sorted(pp):
   rel=(plain[unit]['rmse']-pp[unit]['rmse'])/plain[unit]['rmse'];rows.append({'cohort':cohort,'unit':unit,'pp_rmse':pp[unit]['rmse'],'plain_rmse':plain[unit]['rmse'],'relative_rmse_reduction':rel,'pp_wins':bool(rel>0)})
 values=np.asarray([r['relative_rmse_reduction'] for r in rows]);rng=np.random.default_rng(271828);boot=values[rng.integers(0,len(values),size=(50000,len(values)))].mean(1);wins=int((values>0).sum());result={'status':'descriptive sequential unit-level audit; improved v3 model is post-confirmation development','n_unique_units':len(rows),'pp_unit_wins':wins,'mean_relative_rmse_reduction':float(values.mean()),'median_relative_rmse_reduction':float(np.median(values)),'unit_bootstrap_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'exact_one_sided_sign_test_p':float(binomtest(wins,len(rows),.5,alternative='greater').pvalue),'units':rows,'warning':'Four units are insufficient for a conventional significance claim; temporal rows are not treated as independent samples.'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
