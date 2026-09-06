#!/usr/bin/env python3
"""Label-preserving post-hoc diagnostics for the sealed MATR 2019 result."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matr_2019_latent_confirmatory import MAT,load_cells,make_rows
ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'results/matr_2019_latent_confirmatory/results.json';OUT=ROOT/'results/matr_2019_posthoc_analysis'
def main():
 r=json.load(open(IN));a=r['pretest'];cells,_,_=load_cells();by_id={c['index']:c for c in cells};tc=[by_id[i] for i in a['split_indices']['train']];sc=[by_id[i] for i in a['split_indices']['test']];boundary=a['actual_boundary'];tr=make_rows(tc,boundary,train=True);te=make_rows(sc,boundary,targets=False)
 center=np.median(tr['x'],axis=0);scale=np.quantile(tr['x'],.75,axis=0)-np.quantile(tr['x'],.25,axis=0);scale=np.where(scale>1e-8,scale,1.);tree=cKDTree((tr['x']-center)/scale);nearest=tree.query((te['x']-center)/scale,k=1)[0]
 methods=['ridge','plain','pp','jacobian','latent'];per={m:(r[m] if m=='ridge' else r[m]['ensemble'])['per_unit'] for m in methods};rows=[]
 for group in sorted(per['latent']):
  mask=te['groups']==group;coord=te['coordinate'][mask];idx=int(group[1:]);row={'unit':group,'n':int(mask.sum()),'capacity_distance_median':float((boundary-coord).mean()/np.std(tr['coordinate'])),'feature_nn_distance_median':float(np.median(nearest[mask])),'recent_rate_shift':float(np.median(np.abs((te['x'][mask,1]-center[1])/scale[1])))}
  for m in methods:row[m+'_r2']=per[m][group]['r2'];row[m+'_rmse']=per[m][group]['rmse']
  row['latent_rmse_gain_vs_plain']=row['plain_rmse']-row['latent_rmse'];rows.append(row)
 rng=np.random.default_rng(20260906);comparisons={}
 for base in ['ridge','plain','pp','jacobian']:
  d=np.asarray([x['latent_rmse']-x[base+'_rmse'] for x in rows]);boot=np.mean(rng.choice(d,(50000,len(d)),replace=True),axis=1);comparisons['latent_vs_'+base]={'mean_paired_unit_rmse_delta':float(d.mean()),'median_delta':float(np.median(d)),'units_latent_better':int(np.sum(d<0)),'units_total':len(d),'bootstrap_95_ci':[float(v) for v in np.quantile(boot,[.025,.975])]}
 correlations={}
 gain=np.asarray([x['latent_rmse_gain_vs_plain'] for x in rows])
 for key in ['capacity_distance_median','feature_nn_distance_median','recent_rate_shift']:
  stat=spearmanr([x[key] for x in rows],gain);correlations[key]={'spearman_gain_vs_plain':float(stat.statistic),'p_value_descriptive':float(stat.pvalue)}
 payload={'status':'post-hoc diagnostic; no model or threshold selection','source_result':str(IN.relative_to(ROOT)),'per_unit':rows,'paired_unit_bootstrap':comparisons,'prelabel_covariate_correlations':correlations,'warning':'n=10; correlations are exploratory and are not an applicability certificate'};OUT.mkdir(exist_ok=True);(OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 with (OUT/'per_unit.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(10,3.8));units=[x['unit'] for x in rows];delta=np.asarray([x['latent_rmse']-x['plain_rmse'] for x in rows]);ax[0].bar(units,delta,color=np.where(delta<0,'#2474B5','#C84B31'));ax[0].axhline(0,color='black',lw=.8);ax[0].set_ylabel('Latent PP RMSE − plain NN RMSE');ax[0].set_title('Per-cell paired error difference')
 distance=np.asarray([x['feature_nn_distance_median'] for x in rows]);ax[1].scatter(distance,-delta,c=np.where(delta<0,'#2474B5','#C84B31'),s=45);ax[1].axhline(0,color='black',lw=.8);ax[1].set_xlabel('Median feature nearest-neighbor distance');ax[1].set_ylabel('Latent PP RMSE gain');ax[1].set_title('Distance does not explain gain');fig.tight_layout();fig.savefig(OUT/'per_unit_diagnostic.png',dpi=180);plt.close(fig)
 print(json.dumps({'bootstrap':comparisons,'correlations':correlations},indent=2))
if __name__=='__main__':main()
