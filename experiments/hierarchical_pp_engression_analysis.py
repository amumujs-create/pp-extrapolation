#!/usr/bin/env python3
"""Hierarchical physical-unit analysis of final PP against official Engression."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
SOURCES={
'hust':('results/hust_regime_transport_pp_v1/results.json',['transported']),
'virkler':('results/support_gated_cross_domain_v1/results.json',['datasets','virkler','ensembles','pp']),
'sunwoda':('results/output_calibration_cross_dataset_v1/results.json',['datasets','sunwoda','final','ensemble']),
'rwth':('results/output_calibration_cross_dataset_v1/results.json',['datasets','rwth','final','ensemble']),
'matr':('results/matr_pp_validation_calibration_v1/results.json',['final_ensemble']),
'matr_batch2':('results/matr_batch2_pp_equal_tuning_v1/results.json',['transported']),
'nasa':('results/nasa_causal_multiscale_pp_v1/results.json',['ensemble'])}
def node(file,path):
 d=json.load(open(ROOT/file))
 for p in path:d=d[p]
 return d
def ncmapss_pp():
 from ncmapss_tra_quantile_split import make_tra_hard_split
 split=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);aw=split.all_windows
 mask=aw.hard_extrap_mask(split.meta['unit_ids']['test'],thresholds=split.thresholds)
 pred=np.mean([np.load(ROOT/f'results/ncmapss_pp_multiscale_v1/seed{s}.npz')['prediction'] for s in range(42,47)],0);out={}
 for u in np.unique(aw.units[mask]):
  m=mask&(aw.units==u);out[str(int(u))]={'rmse':float(np.sqrt(np.mean((pred[m]-aw.y[m])**2))),'n':int(m.sum())}
 return {'per_unit':out}
def main():
 en=json.load(open(ROOT/'results/engression_all_positive_v2/results.json'))['datasets']; rng=np.random.default_rng(20260907); draws=100000; rows={}; rels=[]
 pp={k:node(*v) for k,v in SOURCES.items()};pp['ncmapss']=ncmapss_pp()
 for ds in pp:
  common_all=sorted(set(pp[ds]['per_unit'])&set(en[ds]['ensemble']['per_unit'])); common=[u for u in common_all if min(pp[ds]['per_unit'][u]['n'],en[ds]['ensemble']['per_unit'][u]['n']) >= 2]; vals=[];details={}
  for u in common:
   a=pp[ds]['per_unit'][u];b=en[ds]['ensemble']['per_unit'][u];rel=(a['rmse']-b['rmse'])/b['rmse'];vals.append(rel);details[u]={'n':a['n'],'pp_rmse':a['rmse'],'engression_rmse':b['rmse'],'relative_delta':rel}
  v=np.asarray(vals);boot=v[rng.integers(0,len(v),(draws,len(v)))].mean(1);rels.append(v)
  rows[ds]={'minimum_observations_per_unit':2,'excluded_units_below_minimum':sorted(set(common_all)-set(common)),'n_units':len(v),'pp_better_units':int((v<0).sum()),'mean_relative_rmse_delta':float(v.mean()),'median_relative_rmse_delta':float(np.median(v)),'unit_bootstrap_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'details':details}
 # Each draw resamples datasets, then physical units within selected datasets; datasets receive equal weight.
 h=np.empty(draws)
 keys=list(rows)
 for i in range(draws):
  chosen=rng.integers(0,len(keys),len(keys)); means=[]
  for j in chosen:
   v=rels[j];means.append(v[rng.integers(0,len(v),len(v))].mean())
  h[i]=np.mean(means)
 allv=np.concatenate(rels); wins=int((allv<0).sum())
 lodo={}
 for drop in range(len(keys)):
  keep=[j for j in range(len(keys)) if j!=drop]; z=np.empty(50000)
  for i in range(len(z)):
   chosen=rng.choice(keep,size=len(keep),replace=True); means=[]
   for j in chosen:
    v=rels[j]; means.append(v[rng.integers(0,len(v),len(v))].mean())
   z[i]=np.mean(means)
  lodo[keys[drop]]={'mean_relative_rmse_delta':float(np.mean([rels[j].mean() for j in keep])),'hierarchical_ci95':[float(x) for x in np.quantile(z,[.025,.975])]}
 result={'status':'post-hoc hierarchical audit; relative RMSE makes heterogeneous target scales comparable; primary physical-unit analysis requires n>=2 so a unit error is not defined by a single point','datasets':rows,'summary':{'n_datasets':len(rows),'n_physical_units':len(allv),'pp_better_units':wins,'unit_win_rate':wins/len(allv),'exact_unit_sign_test_p_descriptive':float(binomtest(wins,len(allv),.5,alternative='greater').pvalue),'equal_dataset_mean_relative_rmse_delta':float(np.mean([v.mean() for v in rels])),'hierarchical_bootstrap_ci95':[float(x) for x in np.quantile(h,[.025,.975])],'bootstrap_probability_below_zero':float((h<0).mean()),'leave_one_dataset_out':lodo,'warning':'Post-hoc; units within datasets and datasets themselves are not guaranteed exchangeable. This strengthens retrospective evidence but cannot create prospective confirmation.'}}
 out=ROOT/'results/hierarchical_pp_engression_v1';out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2));print('\n'.join(f'{k}: {v["pp_better_units"]}/{v["n_units"]}, mean={v["mean_relative_rmse_delta"]:.3f}, CI={v["unit_bootstrap_ci95"]}' for k,v in rows.items()))
if __name__=='__main__':main()
