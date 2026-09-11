#!/usr/bin/env python3
"""Paper figure for retrospective PP-X policy selection audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ppx_paper_policy_audit_v1/results.json"
OUT = ROOT / "figures/ppx_paper_policy_v1"


def main():
    payload = json.loads(SOURCE.read_text())
    policies = payload["policies"]
    order = [
        "always_direct",
        "always_pp",
        "simple_validation",
        "paper_ppx",
        "test_oracle_non_deployable",
    ]
    labels = [
        "Always\ndirect",
        "Always\nPP",
        "Validation\nRMSE only",
        "Frozen\nPP-X",
        "Test oracle\n(non-deployable)",
    ]
    correct = [policies[name]["correct"] for name in order]
    false_accepts = [policies[name]["false_accepts"] for name in order]
    false_rejects = [policies[name]["false_rejects"] for name in order]
    colors = ["#777777", "#9B1C1C", "#E69F00", "#0072B2", "#009E73"]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    axes[0].bar(x, correct, color=colors)
    axes[0].axhline(6, color="#333333", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0, 12.8)
    axes[0].set_ylabel("Correct route decisions / 12")
    axes[0].set_title("Retrospective route accuracy", loc="left", fontweight="bold")
    for i, value in enumerate(correct):
        axes[0].text(i, value + 0.25, str(value), ha="center")

    axes[1].bar(x, false_accepts, color=colors, label="False accepts")
    axes[1].bar(
        x,
        false_rejects,
        bottom=false_accepts,
        color="none",
        edgecolor=colors,
        hatch="//",
        linewidth=1.2,
        label="False rejects",
    )
    axes[1].set_ylim(0, 12.8)
    axes[1].set_ylabel("Route errors / 12")
    axes[1].set_title("Error decomposition", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E8E8E8")
        ax.set_axisbelow(True)
    fig.suptitle(
        "Validation alone is insufficient; typed contracts remain necessary",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"fig_ppx_policy_audit.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    print("saved", OUT)


if __name__ == "__main__":
    main()
