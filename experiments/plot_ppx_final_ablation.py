"""Generate publication-ready PNG/PDF figures for final PP-X ablations."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ppx_final_ablation_statistics_v1/results.json"
OUT = ROOT / "figures/paper/ppx_final_ablation"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

def save(fig, name):
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    rows = json.loads(SOURCE.read_text())["comparisons"]

    # Complete component effect overview.
    labels = [f'{r["dataset"]}\n{r["component"]}' for r in rows]
    delta = np.asarray([r["ensemble_delta_r2"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(rows)); colors = np.where(delta >= 0, "#7B001C", "#777777")
    ax.barh(y, delta, color=colors); ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("Prediction-ensemble Δ pooled R² (on − off)")
    ax.set_title("Final PP-X component ablation")
    save(fig, "fig_ablation_delta_r2")

    # Unit-paired forest plot.
    mean = np.asarray([r["mean_unit_rmse_reduction"] for r in rows])
    ci = np.asarray([r["unit_bootstrap_ci95"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(mean, y, xerr=np.vstack((mean-ci[:, 0], ci[:, 1]-mean)), fmt="o",
                color="#7B001C", ecolor="#999999", capsize=2)
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel("Unit RMSE reduction (off − on), bootstrap 95% CI")
    ax.set_title("Physical-unit paired effects")
    save(fig, "fig_ablation_unit_forest")

    # Total selected PP-X against direct NN.
    total = [r for r in rows if r["component"] == "complete selected PP-X vs direct NN"]
    names = [r["dataset"] for r in total]; off = [r["off_ensemble_r2"] for r in total]; on = [r["on_ensemble_r2"] for r in total]
    x = np.arange(len(total)); w=.36
    fig, ax = plt.subplots(figsize=(6.8, 4.2)); ax.bar(x-w/2, off, w, label="Direct NN", color="#999999")
    ax.bar(x+w/2, on, w, label="Selected PP-X", color="#7B001C"); ax.axhline(0,color="black",lw=.8)
    ax.set_xticks(x,names); ax.set_ylabel("Pooled R²"); ax.set_title("Complete selected PP-X versus matched direct NN"); ax.legend(frameon=False)
    save(fig, "fig_selected_ppx_vs_direct")

    # Seed effects for every comparison; shows optimizer stability separately.
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, r in enumerate(rows):
        values=np.asarray(r["seed_delta_r2"]); ax.scatter(values, np.full(len(values), i), s=15, alpha=.65, color="#7B001C")
        ax.plot([values.mean()-.01, values.mean()+.01], [i,i], color="black", lw=2)
    ax.axvline(0,color="black",lw=.8); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("Per-seed Δ pooled R² (on − off)")
    ax.set_title("Retraining-seed sensitivity")
    save(fig, "fig_ablation_seed_stability")

    # Multiple-testing-aware classification of every ablation comparison.
    status=[]
    for r in rows:
        lo,hi=r["unit_bootstrap_ci95"]; q=r["unit_signflip_q_bh"]
        status.append(1 if lo>0 and q<.05 else -1 if hi<0 and q<.05 else 0)
    fig, ax = plt.subplots(figsize=(10, 7)); palette={1:"#177245",0:"#B8B8B8",-1:"#B22222"}
    ax.barh(y, np.ones(len(rows)), color=[palette[s] for s in status])
    ax.set_xlim(0,1); ax.set_xticks([]); ax.set_yticks(y,labels); ax.invert_yaxis()
    ax.set_title("Ablation inference after BH correction")
    for i,s in enumerate(status): ax.text(.02,i,{1:"significant improvement",0:"inconclusive",-1:"significant harm"}[s],va="center",color="white" if s else "black")
    save(fig,"fig_ablation_significance_status")

    # Full exact-split competitor matrix; missing means that method was not run.
    competitors=["PP-X","V-REx","GroupDRO","Monotone NN","Linear-tail RBF","Engression","Linear-mean GP"]
    matrix_rows={
      "HUST":[.958,.809,.934,.822,.710,.878,-.320],
      "Virkler":[.888,.583,.554,.565,.805,.552,.539],
      "NASA battery":[.584,.285,.286,.283,.550,.549,.438],
      "Sunwoda":[.939,-.240,-.295,-.048,.838,.619,-1.598],
      "RWTH":[.878,.645,.602,-.005,.385,.526,-.474],
      "MATR2019":[.466,.044,.272,.018,-2.639,-.726,-2.461],
      "MATR batch 2":[.862,.850,.777,.674,-.781,.739,.213],
      "N-CMAPSS":[.937,.883,.880,.892,.819,.932,.804],
      "MICH":[.751,np.nan,np.nan,-.743,np.nan,np.nan,np.nan],
      "NASA milling":[.341,-.693,-.691,-.694,-5.681,np.nan,np.nan],
    }
    ds=list(matrix_rows); mat=np.asarray(list(matrix_rows.values()))
    fig,ax=plt.subplots(figsize=(10.2,6.2)); shown=np.ma.masked_invalid(mat)
    im=ax.imshow(shown,aspect="auto",cmap="RdYlGn",vmin=-1,vmax=1)
    ax.set_xticks(np.arange(len(competitors)),competitors,rotation=30,ha="right"); ax.set_yticks(np.arange(len(ds)),ds)
    for i in range(len(ds)):
        for j in range(len(competitors)):
            ax.text(j,i,"—" if np.isnan(mat[i,j]) else f"{mat[i,j]:.3f}",ha="center",va="center",fontsize=7.5,
                    color="white" if np.isfinite(mat[i,j]) and abs(mat[i,j])>.65 else "black")
    ax.set_title("Exact-split extrapolation benchmark (prediction-ensemble pooled R²)")
    fig.colorbar(im,ax=ax,label="Pooled R²",shrink=.8); save(fig,"fig_competitor_score_matrix")

    # PP-X versus the strongest available exact-split comparator.
    best=np.nanmax(mat[:,1:],axis=1); pp=mat[:,0]; x=np.arange(len(ds)); w=.36
    fig,ax=plt.subplots(figsize=(9.5,4.8)); ax.bar(x-w/2,pp,w,label="Final selected PP-X",color="#7B001C")
    ax.bar(x+w/2,best,w,label="Strongest available comparator",color="#888888"); ax.axhline(0,color="black",lw=.8)
    ax.set_xticks(x,ds,rotation=35,ha="right"); ax.set_ylabel("Pooled R²"); ax.set_title("PP-X versus strongest available exact-split comparator")
    ax.legend(frameon=False); save(fig,"fig_ppx_vs_strongest_competitor")

    manifest={"source":str(SOURCE.relative_to(ROOT)),"figures":[p.name for p in sorted(OUT.glob("*"))],
              "note":"PNG 300 dpi and vector PDF; generated from stored row-aligned predictions"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__ == "__main__": main()
