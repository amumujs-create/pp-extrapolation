#!/usr/bin/env python3
"""Generate CCMR ablation / stats / benchmark figures for the research PPT."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/ccmr_model_ablation_stats_benchmark_v1/results.json"
OUT = Path("/Users/baghyeongbae/Desktop/연구/ppt/pp/_build/figs")
INK = "#222222"
BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#9B1C1C"
GREY = "#777777"
GREEN = "#009E73"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_dev_v22():
    payload = json.loads(DATA.read_text())
    rows = payload["development_v20_vs_v22"]["domains"]
    names = [row["domain"].replace("_", "\n") for row in rows]
    v20 = [100 * row["v20_pooled_improvement"] for row in rows]
    v22 = [100 * row["v22_pooled_improvement"] for row in rows]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(x - width / 2, v20, width, label="CCMR v2.0", color=GREY)
    ax.bar(x + width / 2, v22, width, label="CCMR v2.2", color=BLUE)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Pooled RMSE improvement (%)")
    ax.set_title("Development domains: v2.0 vs v2.2", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False)
    style(ax)
    save(fig, "ccmr_v22_dev_improvement.png")


def fig_expert_ablation():
    payload = json.loads(DATA.read_text())
    variants = [
        "linear_rate",
        "damped_acceleration",
        "monotone_hinge",
        "nonlinear_residual",
        "uniform_bank",
        "selected_bank",
    ]
    labels = [
        "linear\nrate",
        "damped\naccel.",
        "monotone\nhinge",
        "nonlinear\nresidual",
        "uniform\nbank",
        "selected\nbank",
    ]
    regrets = [
        payload["expert_ablation_v20"]["by_variant"][name]["maximum_raw_regret"]
        for name in variants
    ]
    pos = [
        payload["expert_ablation_v20"]["by_variant"][name]["domains_positive"]
        for name in variants
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    colors = [GREY] * 5 + [BLUE]
    bars = ax.bar(x, regrets, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Maximum raw unit regret")
    ax.set_title(
        "Causal dynamics bank expert ablation (v2.0) — lower regret is better",
        loc="left",
        color=INK,
        fontweight="bold",
    )
    for xi, bar, npos in zip(x, bars, pos):
        ax.text(
            xi,
            bar.get_height() + 0.25,
            f"+domains {npos}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    style(ax)
    save(fig, "ccmr_expert_ablation.png")


def fig_holdout_benchmark():
    payload = json.loads(DATA.read_text())
    order = [
        "CCMR_v2.2",
        "Engression",
        "Linear-mean GP",
        "Linear-tail RBF",
        "CCMR_sealed_success",
        "Persistence",
        "TabPFN v3",
        "Monotone NN",
        "V-REx",
        "GroupDRO",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, cohort, title in zip(
        axes,
        ("Alloy_A", "MultiStage_RPT"),
        ("Alloy A", "MultiStage RPT"),
    ):
        models = payload["holdout_benchmark"]["cohorts"][cohort]["models"]
        labels = []
        r2 = []
        regret = []
        for name in order:
            if name not in models:
                continue
            labels.append(name.replace("_", "\n"))
            r2.append(models[name]["pooled_r2"])
            regret.append(100 * models[name]["max_regret"])
        y = np.arange(len(labels))
        colors = [BLUE if "CCMR" in label.replace("\n", "_") else GREY for label in labels]
        colors = [
            BLUE if "CCMR v2.2" in label.replace("\n", " ")
            else ORANGE if "Engression" in label
            else GREY
            for label in labels
        ]
        ax.barh(y, r2, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels([label.replace("\n", " ") for label in labels], fontsize=8)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_xlabel("Pooled $R^2$")
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
        style(ax)
        for yi, value, reg in zip(y, r2, regret):
            ax.text(
                value + 0.02 if value >= 0 else value - 0.02,
                yi,
                f"reg {reg:.0f}%",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7,
                color=RED if reg > 2 else GREEN if reg <= 0 else GREY,
            )
    fig.suptitle(
        "Holdout competitors with CCMR v2.2 overlay (retrospective)",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "ccmr_holdout_benchmark.png")


def fig_v23_path():
    payload = json.loads(DATA.read_text())
    attempts = payload["v23_rejection"]["attempts"]
    labels = [item["variant"].replace("_", "\n") for item in attempts]
    ratios = [100 * (item["geometric_rmse_ratio_to_v22"] - 1.0) for item in attempts]
    regrets = [100 * item["maximum_test_raw_regret"] for item in attempts]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(10.0, 3.8))
    ax1.plot(x, ratios, "-o", color=BLUE, label="GM RMSE vs v2.2 (%)")
    ax1.axhline(0, color=INK, linewidth=0.8)
    ax1.set_ylabel("GM RMSE change vs v2.2 (%)", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(x, regrets, "--s", color=RED, label="Max raw regret (%)")
    ax2.set_ylabel("Max raw unit regret (%)", color=RED)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_title(
        "CCMR v2.3 AC-CRPE development path (rejected)",
        loc="left",
        color=INK,
        fontweight="bold",
    )
    style(ax1)
    ax2.spines["top"].set_visible(False)
    save(fig, "ccmr_v23_rejection_path.png")


def fig_claim_summary():
    payload = json.loads(DATA.read_text())
    alloy = payload["holdout_benchmark"]["cohorts"]["Alloy_A"]["models"]
    multi = payload["holdout_benchmark"]["cohorts"]["MultiStage_RPT"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    # Holdout replay uses sealed Engression; competitor Engression is the same family.
    eng_key = "Engression_holdout_replay" if "Engression_holdout_replay" in alloy else "Engression"
    for ax, models, title in zip(
        axes,
        (alloy, multi),
        ("Alloy A: accuracy", "MultiStage: safety"),
    ):
        names = ["CCMR_v2.2", eng_key]
        display = ["CCMR v2.2", "Engression"]
        if "accuracy" in title:
            values = [models[name]["pooled_r2"] for name in names]
            ylabel = "Pooled $R^2$"
        else:
            values = [100 * models[name]["max_regret"] for name in names]
            ylabel = "Max raw regret (%)"
        ax.bar(display, values, color=[BLUE, ORANGE])
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
        ax.set_ylabel(ylabel)
        style(ax)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(
        "What CCMR can claim on the two opened holdouts",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "ccmr_claim_summary.png")


def main():
    fig_dev_v22()
    fig_expert_ablation()
    fig_holdout_benchmark()
    fig_v23_path()
    fig_claim_summary()
    print("done", OUT)


if __name__ == "__main__":
    main()
