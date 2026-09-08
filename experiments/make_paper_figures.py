#!/usr/bin/env python3
"""Create publication figures for the final modular PP study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
COLORS = {"pp": "#9E1B32", "base": "#3B6FB6", "neutral": "#697386",
          "good": "#14866D", "bad": "#C54B4B", "gold": "#C69214"}


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def metric_r2(obj):
    return obj.get("r2", obj.get("pooled_r2"))


def figure_1_overall():
    names = ["HUST", "Virkler", "NASA batt.", "Sunwoda", "RWTH", "MATR2019",
             "MATR-b2", "N-CMAPSS", "MICH", "XJTU", "FEMTO", "NASA mill."]
    pp = np.array([.958, .888, .584, .934, .842, .466, .862, .937, .751, -1.229, -1.378, -4.826])
    comp = np.array([.934, .805, .550, .838, .645, .344, .850, .932, .684, -1.418, -.973, -.691])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hlines(y, comp, pp, color="#CBD2DA", lw=2, zorder=1)
    ax.scatter(comp, y, color=COLORS["neutral"], s=28, label="Strongest matched comparator", zorder=2)
    ax.scatter(pp, y, color=COLORS["pp"], s=35, label="Final modular PP", zorder=3)
    ax.axvline(0, color="#222", lw=.8, ls="--")
    ax.set(yticks=y, yticklabels=names, xlabel="Pooled $R^2$")
    ax.invert_yaxis(); ax.set_title("Strict extrapolation performance across datasets", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E7E9ED", lw=.7)
    save(fig, "fig1_final_benchmark")


def figure_2_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 3.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")
    def box(x, y, w, h, text, color, sub=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=.08",
                                    facecolor=color, edgecolor="white", lw=1.2))
        ax.text(x+w/2, y+h*.58, text, ha="center", va="center", color="white", weight="bold", fontsize=8)
        if sub: ax.text(x+w/2, y+h*.27, sub, ha="center", va="center", color="white", fontsize=7)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.2, color="#5F6875"))
    box(.2, 1.35, 1.9, 1.0, "Causal history", COLORS["base"], "health, rate, time")
    box(2.8, 2.25, 2.3, .9, "Frozen affine tail", COLORS["neutral"], "stable extrapolator")
    box(2.8, .55, 2.3, .9, "Neural residual", COLORS["good"], "learned deviation")
    box(5.9, .9, 2.5, 1.9, "Support-adaptive\ndual-scale gate", COLORS["pp"], "local bound ↔ broad bound")
    box(9.2, 1.35, 2.2, 1.0, "Boundary quotient", COLORS["gold"], "margin × positive tail")
    box(12.2, 1.35, 1.6, 1.0, "RUL output", COLORS["base"], "zero at EOL")
    arrow(2.1, 1.85, 2.8, 2.65); arrow(2.1, 1.75, 2.8, 1.0)
    arrow(5.1, 2.65, 5.9, 2.25); arrow(5.1, 1.0, 5.9, 1.4)
    arrow(8.4, 1.85, 9.2, 1.85); arrow(11.4, 1.85, 12.2, 1.85)
    ax.text(7.0, .28, "Validation evidence approves optional history, decay, and transport executors",
            ha="center", va="center", fontsize=8, color="#4E5661")
    ax.set_title("Modular PP: stable tail with support-conditioned neural correction", loc="left", weight="bold")
    save(fig, "fig2_model_overview")


def figure_3_bq_arms():
    d = json.load(open(ROOT / "results/bq_pp_matched_controls_v1/results.json"))
    arms = ["direct_nn", "soft_boundary_nn", "hard_trainable_nn", "affine_quotient_only",
            "frozen_unbounded_pp", "bq_pp"]
    labels = ["Direct NN", "Soft boundary", "Trainable hard BC", "Affine quotient",
              "Frozen + unbounded", "Frozen + bounded"]
    datasets = ["sunwoda", "rwth", "mich"]
    values = np.array([[d["arms"][a]["ensemble"][s]["pooled_r2"] for a in arms] for s in datasets])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), sharey=True)
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        colors = [COLORS["pp"] if a == "bq_pp" else COLORS["base"] for a in arms]
        ax.bar(np.arange(len(arms)), values[i], color=colors, width=.72)
        ax.axhline(0, color="#222", lw=.8); ax.grid(axis="y", color="#ECEEF1", lw=.7)
        ax.set_xticks(np.arange(len(arms)), labels, rotation=55, ha="right")
        ax.set_title(ds.upper(), weight="bold")
    axes[0].set_ylabel("Prediction-ensemble pooled $R^2$")
    fig.suptitle("Matched structural ablation of the boundary-quotient executor", weight="bold", y=1.02)
    fig.tight_layout(); save(fig, "fig3_bq_matched_ablation")


def figure_4_component_heatmap():
    rows = ["Nonlinear residual", "Frozen affine", "Fixed residual bound", "Dual-scale capacity",
            "Causal rate history", "Regime transport"]
    cols = ["Sunwoda", "RWTH", "MICH", "HUST", "MATR-b2"]
    z = np.full((len(rows), len(cols)), np.nan)
    z[0, :3] = [.658, .219, 3.811]; z[1, :3] = [.039, .023, .149]
    z[2, :3] = [.221, .090, -.291]; z[3, :3] = [-.005, -.036, .283]
    z[4, :3] = [.349, 1.256, -.247]; z[5, 3:] = [.128, .187]
    fig, ax = plt.subplots(figsize=(7.2, 3.7)); masked = np.ma.masked_invalid(z)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i,j]:+.3f}", ha="center", va="center",
                        color="white" if abs(z[i,j]) > .24 else "#20242A", fontsize=8, weight="bold")
            else: ax.text(j, i, "N/A", ha="center", va="center", color="#8B929C", fontsize=7)
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows)
    ax.set_title(r"Incremental contribution of PP components ($\Delta R^2$)", loc="left", weight="bold")
    cbar=fig.colorbar(im, ax=ax, fraction=.035, pad=.03); cbar.set_label("With component − matched removal")
    fig.tight_layout(); save(fig, "fig4_component_delta_heatmap")


def figure_5_matr_interaction():
    d = json.load(open(ROOT / "results/matr_batch2_support_transport_ablation_v1/results.json"))["arms"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for key, label, color in [("decay_0.00", "Support decay off", COLORS["base"]),
                              ("decay_0.05", "Support decay on", COLORS["pp"])]:
        vals=[]; errs=[]
        for metric in ["raw", "transported"]:
            x=np.array([r[metric]["pooled"]["r2"] for r in d[key]["runs"]]); vals.append(x.mean()); errs.append(x.std(ddof=1))
        ax.errorbar([0,1], vals, yerr=errs, marker="o", ms=6, lw=2, capsize=4, color=color, label=label)
    ax.set_xticks([0,1],["Transport off","Transport on"]); ax.set_ylabel("Single-seed pooled $R^2$ (mean ± SD)")
    ax.set_title("Support decay and regime transport interaction", loc="left", weight="bold")
    ax.grid(axis="y", color="#E7E9ED"); ax.legend(frameon=False)
    save(fig, "fig5_matr_support_transport_interaction")


def figure_6_seed_stability():
    pp=json.load(open(ROOT/'results/matr_batch2_pp_five_seed_replay/results.json'))
    cg=json.load(open(ROOT/'results/batterylife_strict_tail_matrb2_final_pp/results.json'))['datasets']['matrb2_final']['cpgru']
    ct=json.load(open(ROOT/'results/batterylife_strict_tail_matrb2_final_cptransformer/results.json'))['datasets']['matrb2_final']['cptransformer']
    tab=json.load(open(ROOT/'results/tabpfn_external_batteries_v1/results.json'))['datasets']['matrb2']
    vr=json.load(open(ROOT/'results/equal_budget_competitors_v1/results.json'))['datasets']['matr_batch2']['vrex']
    data=[np.array([r['transported']['pooled']['r2'] for r in pp['runs']]),
          np.array([r['metrics']['pooled']['r2'] for r in vr['runs']]),
          np.array([r['metrics']['pooled']['r2'] for r in tab['runs']]),
          np.array([r['metrics']['pooled']['r2'] for r in cg['runs']]),
          np.array([r['metrics']['pooled']['r2'] for r in ct['runs']])]
    labels=['PP','V-REx','TabPFN','CPGRU','CPTransformer']
    fig,ax=plt.subplots(figsize=(6.4,3.8)); rng=np.random.default_rng(7)
    for i,x in enumerate(data):
        ax.scatter(i+rng.uniform(-.09,.09,len(x)),x,s=30,color=COLORS['pp'] if i==0 else COLORS['base'],alpha=.9)
        ax.plot([i-.18,i+.18],[x.mean(),x.mean()],color='#111',lw=1.5)
    ax.axhline(0,color='#222',lw=.8,ls='--'); ax.set_xticks(range(len(labels)),labels)
    ax.set_ylabel("Pooled $R^2$ per seed"); ax.set_title("Retraining stability on MATR batch 2",loc='left',weight='bold')
    ax.grid(axis='y',color='#E7E9ED'); save(fig,'fig6_matr_seed_stability')


def figure_7_hull():
    d=json.load(open(ROOT/'results/all_dataset_hull_audit_v1/results.json'))['datasets']
    # NASA is reported fold-wise; aggregate it without hiding cohort variation.
    flat={}
    for name, item in d.items():
        if 'folds' in item:
            medians=[f['test']['distance_median'] for f in item['folds']]
            ratios=[f['test_to_validation_distance_ratio'] for f in item['folds']]
            flat[name]={'median':float(np.median(medians)), 'ratio':float(np.median(ratios))}
        else:
            flat[name]={'median':item['test']['distance_median'],
                        'ratio':item.get('test_to_validation_distance_ratio')}
    # A zero validation hull distance makes the ratio undefined; show these as 0
    # distance with a neutral colour and mark them in the caption/guide.
    order=sorted(flat,key=lambda k:flat[k]['median'])
    med=np.array([flat[k]['median'] for k in order])
    ratio=np.array([flat[k]['ratio'] if flat[k]['ratio'] is not None else np.nan for k in order])
    labels=[k.replace('matr_batch2','MATR-b2').replace('nasa_battery','NASA batt.').replace('_',' ') for k in order]
    fig,ax=plt.subplots(figsize=(7.2,4.5)); y=np.arange(len(order)); valid=np.isfinite(ratio)
    sc=ax.scatter(med[valid],y[valid],c=np.clip(ratio[valid],0,8),s=52,cmap='viridis',edgecolor='white',lw=.5)
    ax.scatter(med[~valid],y[~valid],s=52,marker='x',color=COLORS['neutral'],lw=1.4,label='Undefined ratio (validation distance = 0)')
    ax.set_yticks(y,labels); ax.set_xlabel("Median test distance beyond train hull (train SD)")
    ax.set_title("Extrapolation severity and validation-to-test distance shift",loc='left',weight='bold'); ax.grid(axis='x',color='#E7E9ED')
    c=fig.colorbar(sc,ax=ax,fraction=.035,pad=.03);c.set_label("Test / validation distance ratio (clipped at 8)")
    if (~valid).any(): ax.legend(frameon=False,loc='lower right')
    save(fig,'fig7_hull_extrapolation_severity')


def figure_8_executor_effects():
    labels=["HUST\ntransport","NASA\nhistory","MATR-b2\ndecay+transport","N-CMAPSS\nmultiscale",
            "Sunwoda\ndual scale","RWTH\ndual scale","MICH\ndual scale"]
    delta=np.array([.128,.012,.189,.009,.069,.099,2.273])
    fig,ax=plt.subplots(figsize=(7.2,3.8)); colors=[COLORS['good'] if x>=0 else COLORS['bad'] for x in delta]
    ax.bar(range(len(delta)),delta,color=colors,width=.68);ax.axhline(0,color='#222',lw=.8)
    ax.set_xticks(range(len(labels)),labels);ax.set_ylabel(r"Final module gain ($\Delta R^2$)")
    ax.set_title("Dataset-appropriate PP executors improve different failure modes",loc='left',weight='bold')
    for i,x in enumerate(delta): ax.text(i,x+max(delta)*.025,f"{x:+.3f}",ha='center',va='bottom',fontsize=8)
    ax.grid(axis='y',color='#E7E9ED'); save(fig,'fig8_executor_contributions')


def figure_9_ablation_composite():
    """Compact main-text view of structural and executor evidence."""
    d=json.load(open(ROOT/'results/bq_pp_matched_controls_v1/results.json'))
    arms=['direct_nn','soft_boundary_nn','hard_trainable_nn','affine_quotient_only','frozen_unbounded_pp','bq_pp']
    labels=['Direct\nNN','Soft\nboundary','Trainable\nhard BC','Affine\nquotient','Frozen +\nunbounded','Final\nbounded']
    datasets=['sunwoda','rwth','mich']
    vals=np.array([[d['arms'][a]['ensemble'][s]['pooled_r2'] for a in arms] for s in datasets])
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),gridspec_kw={'width_ratios':[1.3,1]})
    x=np.arange(len(arms)); width=.24
    for i,(ds,color) in enumerate(zip(datasets,[COLORS['base'],COLORS['gold'],COLORS['pp']])):
        axes[0].bar(x+(i-1)*width,vals[i],width,label=ds.upper(),color=color)
    axes[0].axhline(0,color='#222',lw=.8);axes[0].set_xticks(x,labels)
    axes[0].set_ylabel('Prediction-ensemble pooled $R^2$');axes[0].set_title('(a) Matched structural ablation',loc='left',weight='bold')
    axes[0].legend(frameon=False,ncol=3);axes[0].grid(axis='y',color='#E7E9ED')
    effect_labels=['HUST\ntransport','NASA\nhistory','MATR-b2\ngate','N-CMAPSS\nmultiscale','Sunwoda\ndual','RWTH\ndual','MICH\ndual']
    delta=np.array([.128,.012,.189,.009,.069,.099,2.273])
    axes[1].bar(range(len(delta)),delta,color=COLORS['good'],width=.68)
    axes[1].set_xticks(range(len(delta)),effect_labels,rotation=40,ha='right')
    axes[1].set_ylabel(r'Module gain ($\Delta R^2$)');axes[1].set_title('(b) Approved executor gains',loc='left',weight='bold')
    axes[1].grid(axis='y',color='#E7E9ED')
    fig.tight_layout();save(fig,'fig9_ablation_composite')


def figure_10_ablation_coverage():
    """Show where component-level evidence exists, including honest gaps."""
    datasets=['HUST','Virkler','NASA batt.','Sunwoda','RWTH','MATR2019','MATR-b2',
              'N-CMAPSS','MICH','XJTU','FEMTO','NASA mill.']
    modules=['Matched\nstructure','History /\nmultiscale','Support\ndecay','Regime\ntransport',
             'Dual-scale\nbound','Gate stress\ntest']
    # 0: not isolated, 1: component on/off or matched arm, 2: stress/failure evidence.
    z=np.zeros((len(datasets),len(modules)),dtype=int)
    for ds in ['Sunwoda','RWTH','MICH']: z[datasets.index(ds),0]=1
    for ds in ['NASA batt.','Sunwoda','RWTH','MATR2019','N-CMAPSS','MICH']:
        z[datasets.index(ds),1]=1
    z[datasets.index('MATR-b2'),2]=1
    for ds in ['HUST','MATR-b2']: z[datasets.index(ds),3]=1
    for ds in ['Sunwoda','RWTH','MICH']: z[datasets.index(ds),4]=1
    for ds in ['Virkler','Sunwoda','N-CMAPSS','MICH','XJTU']:
        z[datasets.index(ds),5]=2
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(['#EEF1F4',COLORS['good'],COLORS['gold']])
    fig,ax=plt.subplots(figsize=(7.6,5.0));ax.imshow(z,cmap=cmap,vmin=0,vmax=2,aspect='auto')
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j,i,{0:'—',1:'✓',2:'△'}[z[i,j]],ha='center',va='center',
                    color='white' if z[i,j] else '#7D8792',weight='bold',fontsize=10)
    ax.set_xticks(range(len(modules)),modules);ax.set_yticks(range(len(datasets)),datasets)
    ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    ax.set_title('Coverage of component-level PP ablations',loc='left',weight='bold',pad=14)
    handles=[mpl.patches.Patch(color=COLORS['good'],label='Matched/on–off evidence'),
             mpl.patches.Patch(color=COLORS['gold'],label='Stress or failure evidence'),
             mpl.patches.Patch(color='#EEF1F4',label='Not isolated')]
    ax.legend(handles=handles,frameon=False,bbox_to_anchor=(0,-.09),loc='upper left',ncol=3)
    fig.tight_layout();save(fig,'fig10_ablation_coverage')


def figure_11_plain_mlp_inference():
    d=json.load(open(ROOT/'results/plain_mlp_paired_inference_v1/results.json'))['datasets']
    order=['hust','virkler','nasa']; labels=['HUST','Virkler','NASA battery']
    est=np.array([d[k]['mean_rmse_reduction_pp_vs_plain'] for k in order])
    lo=np.array([d[k]['ci95_unit_bootstrap'][0] for k in order]);hi=np.array([d[k]['ci95_unit_bootstrap'][1] for k in order])
    y=np.arange(len(order));fig,ax=plt.subplots(figsize=(6.2,3.1))
    ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt='o',color=COLORS['pp'],ecolor=COLORS['neutral'],capsize=4,lw=1.5)
    ax.axvline(0,color='#222',ls='--',lw=.8);ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.set_xlabel('Unit-mean RMSE reduction: plain MLP − PP')
    ax.set_title('Matched PP versus plain MLP: unit-paired inference',loc='left',weight='bold')
    for i,k in enumerate(order): ax.text(hi[i]+.5,i,f"{d[k]['pp_unit_wins']}/{d[k]['n_units']} wins; p={d[k]['two_sided_paired_signflip_p']:.3g}",va='center',fontsize=8)
    ax.grid(axis='x',color='#E7E9ED');save(fig,'fig11_plain_mlp_paired_inference')


def figure_12_applicability_map():
    d=json.load(open(ROOT/'results/cross_domain_mechanism_v1/results.json'))['datasets']
    externals={
        'NASA/UCF':json.load(open(ROOT/'results/nasa_alt_external_locked_v1/results.json')),
        'CALCE':json.load(open(ROOT/'results/calce_external_locked_v1/results.json')),
        'HNEI':json.load(open(ROOT/'results/hnei_external_locked_v1/results.json')),
    }
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for name,r in d.items():
        x=r['descriptors']['normalized_horizon'];y=r['descriptors']['trajectory_heterogeneity']
        success=r['pp_gain_r2']>.05 and r['pp_ensemble']['pooled']['r2']>0
        ax.scatter(x,y,s=35+5*r['descriptors']['descriptor_units'],color=COLORS['good'] if success else COLORS['neutral'],edgecolor='white',lw=.7)
        ax.annotate(name.replace('_battery',' batt.').replace('matr_batch2','MATR-b2'),(x,y),xytext=(4,3),textcoords='offset points',fontsize=7)
    for i,(label,ext) in enumerate(externals.items()):
        x=ext['descriptors']['normalized_horizon'];y=ext['descriptors']['trajectory_heterogeneity'];success=ext['pp_gain_r2']>0 and ext['pp_ensemble']['pooled']['r2']>0
        ax.scatter(x,y,s=180,marker='*',color=COLORS['good'] if success else COLORS['bad'],edgecolor='#222',lw=.7,label='Locked external cohorts' if i==0 else None)
        ax.annotate(label,(x,y),xytext=(7,(-8 if label=='HNEI' else 3)),textcoords='offset points',fontsize=8,weight='bold')
    ax.axvline(.5,color=COLORS['gold'],ls='--',lw=1);ax.axhline(.05,color=COLORS['gold'],ls='--',lw=1)
    ax.set_xlabel('Median distance beyond training support');ax.set_ylabel('Train-unit degradation-law heterogeneity')
    ax.set_title('Empirical applicability map for the common PP backbone',loc='left',weight='bold')
    ax.margins(x=.06,y=.05);ax.grid(color='#E7E9ED');ax.legend(frameon=False,loc='upper right');save(fig,'fig12_applicability_map')


def figure_13_conformal_coverage():
    d=json.load(open(ROOT/'results/cross_domain_mechanism_v1/results.json'))['datasets']
    ext=[json.load(open(ROOT/f'results/{k}_external_locked_v1/results.json')) for k in ['nasa_alt','calce','hnei']]
    names=list(d)+['NASA/UCF ext.','CALCE ext.','HNEI ext.'];rows=[d[k]['conformal90'] for k in d]+[e['conformal90'] for e in ext]
    cov=np.array([r['test_coverage'] for r in rows]);units=np.array([r['n_calibration_units'] for r in rows])
    fig,ax=plt.subplots(figsize=(7.2,4.2));x=np.arange(len(names));colors=[COLORS['good'] if .85<=c<=.98 else COLORS['bad'] for c in cov]
    ax.scatter(x,cov,s=30+12*units,c=colors,edgecolor='white',lw=.6);ax.axhline(.9,color='#222',ls='--',lw=1,label='90% nominal')
    ax.axhspan(.85,.98,color=COLORS['good'],alpha=.08);ax.set_xticks(x,[n.replace('matr_batch2','MATR-b2').replace('nasa_battery','NASA batt.') for n in names],rotation=50,ha='right')
    ax.set_ylim(0,1.05);ax.set_ylabel('Empirical test coverage');ax.set_title('Support-scaled block-conformal interval audit',loc='left',weight='bold')
    ax.grid(axis='y',color='#E7E9ED');ax.legend(frameon=False);save(fig,'fig13_conformal_coverage')


def figure_14_external_predictions():
    z=np.load(ROOT/'results/nasa_alt_external_locked_v1/predictions.npz',allow_pickle=True);y=z['y'];g=z['groups'];plain=z['plain'].mean(0);pp=z['pp'].mean(0)
    fig,axes=plt.subplots(1,len(np.unique(g)),figsize=(9.2,3.2),sharey=True)
    for ax,u in zip(axes,np.unique(g)):
        m=g==u;o=np.argsort(y[m])[::-1];ax.plot(y[m][o],y[m][o],color='#222',ls='--',lw=1,label='Ideal');ax.plot(y[m][o],plain[m][o],marker='o',color=COLORS['base'],label='Plain MLP');ax.plot(y[m][o],pp[m][o],marker='s',color=COLORS['pp'],label='PP')
        ax.set_title(str(u));ax.set_xlabel('True RUL (days)');ax.grid(color='#E7E9ED')
    axes[0].set_ylabel('Predicted RUL (days)');axes[-1].legend(frameon=False,fontsize=7)
    fig.suptitle('One-shot NASA/UCF external result: PP failure under high heterogeneity',weight='bold',y=1.03);fig.tight_layout();save(fig,'fig14_nasa_alt_external_predictions')


def figure_15_external_comparison():
    labels=['NASA/UCF','CALCE','HNEI'];data=[json.load(open(ROOT/f'results/{k}_external_locked_v1/results.json')) for k in ['nasa_alt','calce','hnei']]
    plain=np.array([d['plain_ensemble']['pooled']['r2'] for d in data]);pp=np.array([d['pp_ensemble']['pooled']['r2'] for d in data]);x=np.arange(len(labels));w=.34
    fig,ax=plt.subplots(figsize=(6.4,3.8));ax.bar(x-w/2,plain,w,color=COLORS['base'],label='Plain MLP');ax.bar(x+w/2,pp,w,color=COLORS['pp'],label='PP backbone')
    ax.axhline(0,color='#222',lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Pooled ensemble $R^2$');ax.set_title('Locked external-cohort evaluation after leakage audit',loc='left',weight='bold');ax.grid(axis='y',color='#E7E9ED');ax.legend(frameon=False)
    for j,v in enumerate(plain):ax.text(j-w/2,v+(.04 if v>=0 else -.08),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    for j,v in enumerate(pp):ax.text(j+w/2,v+(.04 if v>=0 else -.08),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    save(fig,'fig15_external_cohort_comparison')


def figure_16_hnei_predictions():
    z=np.load(ROOT/'results/hnei_external_locked_v1/predictions.npz',allow_pickle=True);y=z['y'];g=z['groups'];plain=z['plain'].mean(0);pp=z['pp'].mean(0);units=np.unique(g)
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.8),sharex=False,sharey=True);axes=np.asarray(axes).ravel()
    for ax,u in zip(axes,units):
        m=g==u;o=np.argsort(y[m])[::-1];ax.plot(y[m][o],y[m][o],color='#222',ls='--',lw=1,label='Ideal');ax.plot(y[m][o],plain[m][o],color=COLORS['base'],lw=1.5,label='Plain MLP');ax.plot(y[m][o],pp[m][o],color=COLORS['pp'],lw=1.5,label='PP')
        ax.set_title(f'Test cell {str(u).rsplit("_",1)[-1]}');ax.set_xlabel('True RUL (cycles)');ax.grid(color='#E7E9ED')
    axes[0].set_ylabel('Predicted RUL (cycles)');axes[2].set_ylabel('Predicted RUL (cycles)');axes[-1].legend(frameon=False,fontsize=7)
    fig.suptitle('HNEI locked external cohort: pooled gain with unit heterogeneity',weight='bold');fig.tight_layout();save(fig,'fig16_hnei_external_predictions')


def figure_17_external_recovery():
    recovery=json.load(open(ROOT/'results/external_failure_pp_recovery_v1/safe_final_results.json'))['datasets']
    hnei=json.load(open(ROOT/'results/hnei_external_locked_v1/results.json'))
    labels=['NASA/UCF','CALCE','HNEI (frozen)']
    plain=np.array([recovery['nasa_alt']['plain']['test']['pooled']['r2'],recovery['calce']['plain']['test']['pooled']['r2'],hnei['plain_ensemble']['pooled']['r2']])
    pp=np.array([recovery['nasa_alt']['pp']['test']['pooled']['r2'],recovery['calce']['pp']['test']['pooled']['r2'],hnei['pp_ensemble']['pooled']['r2']])
    x=np.arange(3);w=.34;fig,ax=plt.subplots(figsize=(6.6,3.8));ax.bar(x-w/2,plain,w,color=COLORS['base'],label='Matched plain NN');ax.bar(x+w/2,pp,w,color=COLORS['pp'],label='PP')
    ax.axhline(0,color='#222',lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Pooled ensemble $R^2$');ax.set_title('Safe-continuation PP recovery on failed external cohorts',loc='left',weight='bold');ax.grid(axis='y',color='#E7E9ED');ax.legend(frameon=False)
    for j,v in enumerate(plain):ax.text(j-w/2,v+(.025 if v>=0 else -.035),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    for j,v in enumerate(pp):ax.text(j+w/2,v+(.025 if v>=0 else -.035),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    save(fig,'fig17_external_failure_recovery')


def main():
    setup(); figure_1_overall(); figure_2_architecture(); figure_3_bq_arms(); figure_4_component_heatmap()
    figure_5_matr_interaction(); figure_6_seed_stability(); figure_7_hull(); figure_8_executor_effects()
    figure_9_ablation_composite(); figure_10_ablation_coverage(); figure_11_plain_mlp_inference()
    figure_12_applicability_map();figure_13_conformal_coverage();figure_14_external_predictions()
    figure_15_external_comparison();figure_16_hnei_predictions()
    figure_17_external_recovery()
    print(f"wrote 17 PDF and 17 PNG figures to {OUT}")


if __name__ == "__main__": main()
