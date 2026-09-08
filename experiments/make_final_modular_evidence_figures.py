#!/usr/bin/env python3
"""Submission figures for the development-final modular PP evidence bundle."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/final_modular_pp_evidence_v1/results.json"
OUT = ROOT / "figures/final_modular_pp_evidence_v1"

# Strongest exact-protocol result already reported in the repository.  Three of
# these do not have row-level prediction artifacts and are therefore excluded
# from the paired-unit forest plot below.
REPORTED = {
    "HUST": (0.934, "GroupDRO"),
    "Virkler": (0.805, "linear-tail RBF"),
    "NASA": (0.550, "linear-tail RBF"),
    "SUNWODA": (0.838, "linear-tail RBF"),
    "RWTH": (0.645, "V-REx"),
    "MICH": (0.684, "direct NN"),
    "MATR2019": (0.377, "calibrated FT"),
    "MATR-b2": (0.850, "V-REx"),
    "N-CMAPSS": (0.932, "Engression"),
}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SOURCE.read_text())
    rows = payload["datasets"]
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})

    names = [x["dataset"] for x in rows]
    pp = np.asarray([x["pp_ensemble_r2"] for x in rows])
    other = np.asarray([REPORTED[n][0] for n in names])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    h = 0.34
    ax.barh(y - h / 2, pp, h, color="#235789", label="Final modular PP")
    ax.barh(y + h / 2, other, h, color="#E0A458", label="Strongest reported comparator")
    for i, (a, b) in enumerate(zip(pp, other)):
        ax.text(max(a, b) + 0.012, i, f"Δ {a-b:+.3f}", va="center", fontsize=9)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Pooled $R^2$")
    ax.set_title("Final modular PP on nine positive extrapolation settings")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=2)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig_F1_final_pp_vs_strongest")

    means = np.asarray([x["mean_unit_log_rmse_ratio"] for x in rows])
    cis = np.asarray([x["unit_log_ratio_bootstrap_ci95"] for x in rows])
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    colors = np.where(cis[:, 0] > 0, "#2A9D8F", "#566573")
    ax.errorbar(means, y, xerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)), fmt="none", ecolor="#6B7280", capsize=4, lw=1.6)
    ax.scatter(means, y, c=colors, s=55, zorder=3)
    for i, x in enumerate(rows):
        ax.text(cis[i, 1] + 0.035, i, f"{x['units_won']}/{x['n_units']} units; q={x['bh_q']:.3g}", va="center", fontsize=9)
    ax.axvline(0, color="#1F2937", lw=1.0)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean log[RMSE(comparator) / RMSE(PP)]")
    ax.set_title("Paired physical-unit effects using stored row-aligned predictions")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig_F2_final_pp_unit_forest")

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    x = np.arange(len(rows))
    pp_mean = np.asarray([d["pp_seed_r2_mean"] for d in rows])
    pp_sd = np.asarray([d["pp_seed_r2_sd"] for d in rows])
    base_mean = np.asarray([d["baseline_seed_r2_mean"] for d in rows])
    base_sd = np.asarray([d["baseline_seed_r2_sd"] for d in rows])
    ax.errorbar(x - 0.10, pp_mean, yerr=pp_sd, fmt="o", color="#235789", capsize=3, label="PP: mean ± seed SD")
    ax.errorbar(x + 0.10, base_mean, yerr=base_sd, fmt="s", color="#E0A458", capsize=3, label="Stored comparator: mean ± seed SD")
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("Single-seed pooled $R^2$")
    ax.set_title("Retraining stability across five matched seeds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_F3_final_pp_seed_stability")
    print(f"saved 3 PNG and 3 PDF figures to {OUT}")


if __name__ == "__main__":
    main()
