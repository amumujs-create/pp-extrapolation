#!/usr/bin/env python3
"""Publication figures for the PP journal evidence audits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = json.loads((ROOT / "results/journal_support_geometry_v1/results.json").read_text())
STATISTICS = json.loads((ROOT / "results/journal_statistical_evidence_v1/results.json").read_text())
ROUTE = json.loads((ROOT / "results/journal_validation_route_v1/results.json").read_text())
OUTPUT = ROOT / "figures" / "journal_evidence_v1"

COLORS = {"1D": "#1F4E79", "2D": "#2A9D8F", "3D": "#E9C46A", "gain": "#C14953"}


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def geometry_sensitivity() -> None:
    rows = []
    for name, value in GEOMETRY["datasets"].items():
        if "support" not in value:
            continue
        support = value["support"]
        rows.append((
            name,
            support["ordered_1d_hull"]["outside_fraction"],
            support["pca_2d_hull"]["outside_fraction"],
            support["pca_3d_hull"]["outside_fraction"],
        ))
    names = [row[0] for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    width = 0.24
    for shift, index, label in ((-width, 1, "1D"), (0, 2, "2D"), (width, 3, "3D")):
        ax.barh(y + shift, [row[index] for row in rows], height=width, label=label, color=COLORS[label])
    ax.set(yticks=y, yticklabels=names, xlim=(0, 1.03), xlabel="Fraction of test rows outside training convex hull")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E7EB")
    ax.legend(frameon=False, title="Support space", loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.11))
    ax.set_title("Hull classification changes with coordinate dimension", loc="left", weight="bold")
    fig.tight_layout()
    save(fig, "fig_J1_hull_dimension_sensitivity")


def support_gain_association() -> None:
    labels = {
        "ordered_1d_hull": "1D hull distance",
        "pca_2d_hull": "PCA-2D hull distance",
        "pca_3d_hull": "PCA-3D hull distance",
        "knn10_density_ratio": "10-NN density ratio",
    }
    rows = GEOMETRY["dataset_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    for ax, (key, label) in zip(axes.ravel(), labels.items()):
        x = np.asarray([row[key] for row in rows])
        y = np.asarray([row["gain"] for row in rows])
        shown_y = np.clip(y, -1.0, 1.0)
        ax.scatter(x, shown_y, s=38, color=COLORS["gain"], edgecolor="white", linewidth=0.6, zorder=3)
        for row, xx, yy in zip(rows, x, y):
            label_text = row["dataset"] if yy >= -1 else f"{row['dataset']} ↓ {yy:.2f}"
            ax.annotate(label_text, (xx, np.clip(yy, -1.0, 1.0)), xytext=(4, 3), textcoords="offset points", fontsize=7)
        result = GEOMETRY["equal_dataset_meta_association"][key]
        ax.axhline(0, color="#6B7280", lw=0.8)
        ax.grid(color="#E5E7EB")
        ax.set_xlabel(label)
        ax.set_ylabel("Mean physical-unit relative RMSE gain")
        ax.set_ylim(-1.12, 1.12)
        ax.set_title(f"ρ={result['rho']:.2f}, permutation p={result['permutation_p']:.3f}", fontsize=10)
    fig.suptitle("Support distance alone does not explain PP benefit", x=0.07, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "fig_J2_support_distance_vs_pp_gain")


def effect_forest() -> None:
    rows = STATISTICS["multiplicity_aware_dataset_tests"]
    names = [row["dataset"] for row in rows]
    mean = np.asarray([row["mean_relative_rmse_gain"] for row in rows])
    lo = np.asarray([row["unit_bootstrap_ci95"][0] for row in rows])
    hi = np.asarray([row["unit_bootstrap_ci95"][1] for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    color = np.where(mean > 0, "#2A9D8F", "#C14953")
    shown_mean = np.clip(mean, -1.0, 1.0)
    shown_lo = np.clip(lo, -1.0, 1.0)
    shown_hi = np.clip(hi, -1.0, 1.0)
    ax.errorbar(shown_mean, y, xerr=[shown_mean - shown_lo, shown_hi - shown_mean], fmt="none", ecolor="#6B7280", capsize=3, lw=1.2)
    ax.scatter(shown_mean, y, c=color, s=45, zorder=3)
    for i, row in enumerate(rows):
        effect = f"; effect={mean[i]:.2f}" if mean[i] < -1 else ""
        ax.text(min(shown_hi[i] + 0.035, 1.02), i, f"{row['unit_wins']}/{row['n_units']}; q={row['bh_fdr_adjusted_p']:.3g}{effect}", va="center", fontsize=7.5)
    ax.axvline(0, color="#111827", lw=0.9)
    ax.set(yticks=y, yticklabels=names, xlabel="Relative RMSE gain of PP over matched MLP")
    ax.set_xlim(-1.08, 1.62)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E7EB")
    ax.set_title("Matched common-backbone effects by physical unit", loc="left", weight="bold")
    fig.tight_layout()
    save(fig, "fig_J3_matched_effect_forest")


def validation_transfer() -> None:
    rows = ROUTE["datasets"]
    x = np.asarray([row["validation_relative_gain"] for row in rows])
    y = np.asarray([row["test_pp_mean_relative_unit_gain"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.7, 6.4))
    shown_y = np.clip(y, -1.0, 1.0)
    ax.scatter(x, shown_y, s=48, color="#6C5CE7", edgecolor="white", linewidth=0.7, zorder=3)
    for row, xx, yy in zip(rows, x, y):
        label_text = row["dataset"] if yy >= -1 else f"{row['dataset']} ↓ {yy:.2f}"
        ax.annotate(label_text, (xx, np.clip(yy, -1.0, 1.0)), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.axvline(0, color="#111827", lw=0.9)
    ax.grid(color="#E5E7EB")
    ax.set(xlabel="Validation relative RMSE gain", ylabel="Test mean physical-unit relative RMSE gain")
    ax.set_ylim(-1.12, 1.12)
    ax.set_title("Validation gain does not reliably transfer across regimes", loc="left", weight="bold")
    fig.tight_layout()
    save(fig, "fig_J4_validation_to_test_transfer")


def main() -> None:
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    geometry_sensitivity()
    support_gain_association()
    effect_forest()
    validation_transfer()
    print(f"saved 4 PNG and 4 PDF figures to {OUTPUT}")


if __name__ == "__main__":
    main()
