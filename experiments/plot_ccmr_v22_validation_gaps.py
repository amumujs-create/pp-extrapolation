#!/usr/bin/env python3
"""Create paper/PPT figures for CCMR v2.2 validation-gap analyses."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results/ccmr_v22_matched_development_benchmark/results.json"
AUDIT = ROOT / "results/ccmr_v22_validation_gap_audit/results.json"
OUT = Path("/Users/baghyeongbae/Desktop/연구/ppt/pp/_build/figs")
BLUE, ORANGE, RED, GREY, INK = (
    "#0072B2", "#E69F00", "#9B1C1C", "#777777", "#222222"
)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8")
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def benchmark():
    data = json.loads(BENCH.read_text())["summary"]
    order = [
        "CCMR_v2.2", "persistence", "validated_portfolio", "ridge",
        "linear_tail_rff", "mlp_ensemble", "engression_ensemble",
    ]
    labels = [
        "CCMR\nv2.2", "Persistence", "Validated\nportfolio", "Ridge",
        "Linear-tail\nRFF", "MLP", "Engression",
    ]
    ratios = [data[name]["geometric_rmse_ratio_to_persistence"] for name in order]
    regrets = [100 * data[name]["maximum_raw_regret"] for name in order]
    colors = [BLUE, GREY, ORANGE, GREY, GREY, GREY, GREY]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    axes[0].bar(x, ratios, color=colors)
    axes[0].axhline(1, color=INK, linewidth=0.8)
    axes[0].set_ylabel("GM RMSE / persistence")
    axes[0].set_title("Accuracy (lower is better)", loc="left", fontweight="bold")
    axes[1].bar(x, regrets, color=colors)
    axes[1].axhline(2, color=RED, linestyle="--", label="2% safety cap")
    axes[1].set_yscale("symlog", linthresh=2)
    axes[1].set_ylabel("Maximum raw unit regret (%)")
    axes[1].set_title("Worst-unit safety", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        style(ax)
    fig.suptitle(
        "Same-split development benchmark: accuracy–safety trade-off",
        x=0.02, ha="left", fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "ccmr_v22_matched_development_benchmark.png")


def threshold():
    rows = json.loads(AUDIT.read_text())["route_threshold_sensitivity"]["rows"]
    active = sorted(set(row["active_fraction_min"] for row in rows))
    small = sorted(set(row["small_macro_gain_min"] for row in rows))
    fig, axes = plt.subplots(1, len(active), figsize=(10.5, 3.4), sharey=True)
    for ax, active_min in zip(axes, active):
        subset = [row for row in rows if row["active_fraction_min"] == active_min]
        stable = sorted(set(row["stable_macro_gain_min"] for row in subset))
        matrix = np.zeros((len(small), len(stable)))
        for i, small_min in enumerate(small):
            for j, stable_min in enumerate(stable):
                row = next(
                    value for value in subset
                    if value["small_macro_gain_min"] == small_min
                    and value["stable_macro_gain_min"] == stable_min
                )
                matrix[i, j] = row["unsafe_accepts"]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="OrRd", aspect="auto")
        ax.set_xticks(range(len(stable)))
        ax.set_xticklabels([f"{100*x:.1f}%" for x in stable])
        ax.set_yticks(range(len(small)))
        ax.set_yticklabels([f"{100*x:.0f}%" for x in small])
        ax.set_xlabel("stable gain minimum")
        ax.set_title(f"active ≥ {active_min:.1f}", fontweight="bold")
        for i in range(len(small)):
            for j in range(len(stable)):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    axes[0].set_ylabel("small gain minimum")
    fig.colorbar(image, ax=axes, label="unsafe accepts", fraction=0.025)
    fig.suptitle(
        "Frozen route threshold sensitivity (diagnostic, no retuning)",
        x=0.02, ha="left", fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.18, wspace=0.25)
    save(fig, "ccmr_v22_route_threshold_sensitivity.png")


def unit_effect():
    rows = json.loads(AUDIT.read_text())["unit_paired_v22_vs_v20"]["datasets"]
    names = [row["domain"].replace("_", "\n") for row in rows]
    means = [100 * row["mean_relative_rmse_gain"] for row in rows]
    lower = [
        means[i] - 100 * row["unit_bootstrap_ci95"][0]
        for i, row in enumerate(rows)
    ]
    upper = [
        100 * row["unit_bootstrap_ci95"][1] - means[i]
        for i, row in enumerate(rows)
    ]
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    x = np.arange(len(rows))
    ax.bar(x, means, color=[BLUE if value > 0 else GREY for value in means])
    ax.errorbar(x, means, yerr=[lower, upper], fmt="none", color=INK, capsize=4)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean unit relative RMSE gain (%)")
    ax.set_title(
        "CCMR v2.2 vs v2.0: physical-unit paired effect",
        loc="left", fontweight="bold",
    )
    style(ax)
    save(fig, "ccmr_v22_unit_paired_effect.png")


if __name__ == "__main__":
    benchmark()
    threshold()
    unit_effect()
