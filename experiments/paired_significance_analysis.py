#!/usr/bin/env python3
"""Post-hoc paired evidence for PP versus Engression without row-wise IID claims."""
from pathlib import Path
import json, sys
import numpy as np
from scipy.stats import binomtest, spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]

def boot(delta, draws=50000, seed=20260907):
    d=np.asarray(delta,float); rng=np.random.default_rng(seed)
    b=d[rng.integers(0,len(d),size=(draws,len(d)))].mean(1)
    return {'n_units':len(d),'mean_delta_rmse_pp_minus_engression':float(d.mean()),
            'median_delta':float(np.median(d)),'pp_better_units':int((d<0).sum()),
            'ci95_percentile':[float(x) for x in np.quantile(b,[.025,.975])],
            'bootstrap_probability_mean_delta_below_zero':float((b<0).mean())}

def per_unit_from_npz():
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_tra_quantile_split import make_tra_hard_split
    split=make_tra_hard_split((ROOT/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42)
    aw=split.all_windows; test_units=split.meta['unit_ids']['test']; mask=aw.hard_extrap_mask(test_units,thresholds=split.thresholds)
    ps=[]
    for seed in range(42,47): ps.append(np.load(ROOT/f'results/ncmapss_pp_multiscale_v1/seed{seed}.npz')['prediction'])
    pred=np.mean(ps,axis=0); out={}
    for u in np.unique(aw.units[mask]):
        m=mask & (aw.units==u); out[str(int(u))]={'rmse':float(np.sqrt(np.mean((pred[m]-aw.y[m])**2))),'n':int(m.sum())}
    return out

def main():
    eng=json.load(open(ROOT/'results/engression_all_positive_v2/results.json'))['datasets']
    nasa_pp=json.load(open(ROOT/'results/nasa_regime_spline_tuning_v1/results.json'))['ensemble']['per_unit']
    nasa_en=eng['nasa']['ensemble']['per_unit']
    ncmp_pp=per_unit_from_npz(); ncmp_en=eng['ncmapss']['ensemble']['per_unit']
    paired={}
    for name,a,b in [('nasa',nasa_pp,nasa_en),('ncmapss',ncmp_pp,ncmp_en)]:
        common=sorted(set(a)&set(b)); delta=[a[u]['rmse']-b[u]['rmse'] for u in common]
        paired[name]={'units':common,'per_unit':{u:{'pp_rmse':a[u]['rmse'],'engression_rmse':b[u]['rmse'],'delta':a[u]['rmse']-b[u]['rmse'],'n':a[u]['n']} for u in common},'bootstrap':boot(delta)}
    pp_runs=[x['raw']['r2'] for x in json.load(open(ROOT/'results/ncmapss_pp_multiscale_v1/results.json'))['runs']]
    en_runs=[x['metrics']['pooled']['r2'] for x in eng['ncmapss']['runs']]
    wins=sum(a>b for a,b in zip(pp_runs,en_runs)); paired['ncmapss']['seed_pairing']={'pp_r2':pp_runs,'engression_r2':en_runs,'wins':wins,'ties':sum(a==b for a,b in zip(pp_runs,en_runs)),'one_sided_exact_binomial_p':float(binomtest(wins,len(pp_runs),.5,alternative='greater').pvalue),'warning':'Same numeric seeds do not align stochastic draws across different algorithms; descriptive robustness check only.'}
    names=['hust','virkler','nasa','sunwoda','rwth','matr','matr_batch2','ncmapss']
    pp=np.array([.958,.888,.571,.865,.743,.466,.862,.937]); en=np.array([eng[n]['ensemble']['pooled']['r2'] for n in names]); dist=np.array([1.608,1.623,2.005,4.421,2.870,5.554,1.898,.228])
    rho,p=spearmanr(dist,pp-en)
    meta={'datasets':names,'pp_minus_engression':(pp-en).tolist(),'wins':int((pp>en).sum()),'one_sided_sign_p_if_prespecified_independent':float(binomtest(int((pp>en).sum()),len(pp),.5,alternative='greater').pvalue),'spearman_hull_distance_vs_gain':float(rho),'spearman_two_sided_p':float(p),'warning':'Exploratory and post-hoc; datasets are heterogeneous and not guaranteed independent.'}
    result={'status':'post-hoc inferential audit; physical-unit bootstrap is preferred over row bootstrap','paired_unit':paired,'cross_dataset_descriptive':meta}
    out=ROOT/'results/paired_pp_engression_v1';out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
