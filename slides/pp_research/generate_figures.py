#!/usr/bin/env python3
"""Generate research figures for the PP briefing deck."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

OUT = Path("/Users/baghyeongbae/Desktop/연구/ppt/pp/_build/figs")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "AppleGothic",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 180,
    }
)

INK = "#102A43"
NAVY = "#163A5F"
CRIMSON = "#8B0029"
BLUE = "#2E6F95"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
GOLD = "#F6BD60"
RED = "#C44536"
GREEN = "#3A7D44"
GREY = "#627D98"
LINE = "#C8D4E3"
PALE = "#F5F8FA"
WASH = "#EAF2F8"


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def fig_motivation():
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # x axis
    ax.plot([0.8, 10.4], [1.4, 1.4], color=NAVY, lw=2.2)
    ax.plot([0.8, 0.8], [1.4, 3.5], color=NAVY, lw=2.2)
    ax.text(0.35, 2.5, "예측", rotation=90, va="center", color=GREY, fontsize=11)
    ax.text(9.7, 1.05, "열화 좌표 / 시간", color=GREY, fontsize=11)

    xs_in = np.linspace(1.0, 5.2, 40)
    y_true = 3.2 - 0.22 * (xs_in - 1) ** 1.15
    ax.plot(xs_in, y_true, color=BLUE, lw=3, label="관측 궤적")
    ax.scatter(xs_in[::5], y_true[::5], color=BLUE, s=28, zorder=3)

    # boundary
    ax.axvline(5.2, color=RED, lw=2, ls="--")
    ax.text(5.3, 3.55, "학습 support 끝", color=RED, fontsize=11, fontweight="bold")

    xs_out = np.linspace(5.2, 10.0, 50)
    # multiple futures
    futures = [
        (3.2 - 0.22 * (5.2 - 1) ** 1.15 - 0.18 * (xs_out - 5.2), TEAL, "가능한 미래 A"),
        (3.2 - 0.22 * (5.2 - 1) ** 1.15 - 0.05 * (xs_out - 5.2) - 0.03 * (xs_out - 5.2) ** 1.4, ORANGE, "가능한 미래 B"),
        (3.2 - 0.22 * (5.2 - 1) ** 1.15 + 0.12 * (xs_out - 5.2) - 0.04 * (xs_out - 5.2) ** 1.2, CRIMSON, "NN 폭주 경로"),
    ]
    for y, c, lab in futures:
        ax.plot(xs_out, y, color=c, lw=2.4, ls="-", alpha=0.95)
        ax.text(9.15, y[-1], lab, color=c, fontsize=10, va="center")

    ax.fill_between([0.8, 5.2], 0.6, 3.7, color=WASH, alpha=0.7, zorder=0)
    ax.text(2.4, 0.85, "관측 가능 구간 (여러 함수가 비슷하게 맞음)", color=NAVY, fontsize=11)
    ax.text(6.6, 0.85, "밖 = 가정이 방향을 결정", color=CRIMSON, fontsize=11, fontweight="bold")
    ax.set_title("연구 모티베이션: support 밖에서는 데이터만으로 함수가 정해지지 않는다", color=INK, fontsize=14, pad=8, loc="left")
    save(fig, "motivation_futures.png")


def fig_prior_ladder():
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    levels = [
        (0.4, "0", "구조 지식 없음", "일반 예측 + 거절", GREY),
        (3.0, "1", "경계·이력", "PP (현재)", TEAL),
        (5.6, "2", "방향·부호", "제약 신경망", BLUE),
        (8.2, "3", "수식·물리", "물리식 결합", ORANGE),
    ]
    for x, lv, name, exe, col in levels:
        box = FancyBboxPatch((x, 1.2), 2.2, 2.4, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=PALE, edgecolor=col, linewidth=2.5)
        ax.add_patch(box)
        circ = Circle((x + 1.1, 3.05), 0.28, facecolor=col, edgecolor="none")
        ax.add_patch(circ)
        ax.text(x + 1.1, 3.05, lv, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        ax.text(x + 1.1, 2.35, name, ha="center", va="center", color=INK, fontsize=13, fontweight="bold")
        ax.text(x + 1.1, 1.7, exe, ha="center", va="center", color=GREY, fontsize=11)
        if x < 8:
            ax.annotate("", xy=(x + 2.45, 2.4), xytext=(x + 2.2, 2.4),
                        arrowprops=dict(arrowstyle="->", color=LINE, lw=2))
    # highlight PP
    ax.add_patch(FancyBboxPatch((2.95, 1.1), 2.3, 2.6, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor="none", edgecolor=TEAL, linewidth=3, linestyle="--"))
    ax.text(0.4, 0.45, "강함 →", color=GREY, fontsize=11)
    ax.text(9.4, 0.45, "← 강한 prior", color=GREY, fontsize=11, ha="right")
    ax.set_title("Prior ladder — 식(강함)은 PAE, 약한 구조 bias는 PP (equation-free safeguard)", color=INK, fontsize=12, loc="left", pad=6)
    save(fig, "prior_ladder.png")


def fig_pp_core_idea():
    """Why plain NN is risky OOS; what PP restricts."""
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.35, 1.55), 3.6, 2.9, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=WASH, edgecolor=TEAL, linewidth=2.2))
    ax.text(2.15, 4.15, "관측된 train 구간", ha="center", color=TEAL, fontsize=13, fontweight="bold")
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.7, 3.5, 18)
    ys = rng.uniform(1.9, 3.6, 18)
    ax.scatter(xs, ys, s=55, color=TEAL, zorder=3, edgecolors="white", linewidths=0.6)

    ax.plot([4.15, 4.15], [1.55, 4.45], color=CRIMSON, lw=2.5)
    ax.annotate("", xy=(10.9, 3.0), xytext=(4.25, 3.0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=2.2))
    ax.text(7.5, 4.25, "관측되지 않은 외삽 구간", ha="center", color=CRIMSON, fontsize=13, fontweight="bold")

    ax.add_patch(FancyBboxPatch((4.5, 3.15), 6.4, 0.85, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="white", edgecolor=RED, linewidth=1.8))
    ax.text(7.7, 3.57, "일반 NN:  자유롭게 연장  →  불안정·폭주 가능",
            ha="center", va="center", color=RED, fontsize=12, fontweight="bold")

    ax.add_patch(FancyBboxPatch((4.5, 1.75), 6.4, 1.15, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="white", edgecolor=TEAL, linewidth=2.2))
    ax.text(7.7, 2.55, "PP:  안정적 추세(affine)를 기본으로,",
            ha="center", va="center", color=TEAL, fontsize=12, fontweight="bold")
    ax.text(7.7, 2.05, "NN은 제한된 보정만 수행",
            ha="center", va="center", color=INK, fontsize=12)

    ax.add_patch(FancyBboxPatch((0.35, 0.3), 10.7, 0.95, boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor=PALE, edgecolor=LINE))
    ax.text(5.7, 0.78,
            "PP는 외삽 구간에서 NN의 자유도를 줄이고,\n데이터가 지지하는 추세를 중심으로 예측을 보정한다.",
            ha="center", va="center", color=INK, fontsize=13, fontweight="bold")
    ax.set_title("PP: support-aware residual extrapolation", color=INK, fontsize=14, loc="left")
    save(fig, "pp_core_idea.png")


def fig_pp_architecture():
    """Affine tail + bounded residual + support gate."""
    fig, ax = plt.subplots(figsize=(11.4, 5.35))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((3.55, 4.35), 4.3, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=WASH, edgecolor=BLUE, linewidth=2))
    ax.text(5.7, 4.7, "history / health / rate / time", ha="center", va="center",
            color=BLUE, fontsize=13, fontweight="bold")

    ax.add_patch(FancyBboxPatch((1.0, 2.55), 3.8, 1.35, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="white", edgecolor=ORANGE, linewidth=2.4))
    ax.text(2.9, 3.55, "Affine tail", ha="center", color=ORANGE, fontsize=14, fontweight="bold")
    ax.text(2.9, 3.0, "stable trend\nglobal extrapolation path", ha="center", color=INK, fontsize=11)

    ax.add_patch(FancyBboxPatch((6.6, 2.55), 3.8, 1.35, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="white", edgecolor=TEAL, linewidth=2.4))
    ax.text(8.5, 3.55, "Residual NN", ha="center", color=TEAL, fontsize=14, fontweight="bold")
    ax.text(8.5, 3.0, "local nonlinear correction\nhistory-driven", ha="center", color=INK, fontsize=11)

    ax.annotate("", xy=(2.9, 3.9), xytext=(4.8, 4.35),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8))
    ax.annotate("", xy=(8.5, 3.9), xytext=(6.6, 4.35),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8))

    ax.add_patch(FancyBboxPatch((2.4, 1.45), 6.6, 0.75, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="white", edgecolor=NAVY, linewidth=2.2))
    ax.text(5.7, 1.82, "support-distance gate  +  bounded residual",
            ha="center", va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.annotate("", xy=(2.9, 2.2), xytext=(2.9, 2.55),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))
    ax.annotate("", xy=(8.5, 2.2), xytext=(8.5, 2.55),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))

    ax.add_patch(FancyBboxPatch((3.9, 0.55), 3.6, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=NAVY, edgecolor="none"))
    ax.text(5.7, 0.85, "RUL prediction", ha="center", va="center", color="white",
            fontsize=13, fontweight="bold")
    ax.annotate("", xy=(5.7, 1.15), xytext=(5.7, 1.45),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))

    ax.set_title("모형 구조: affine 기본 경로 + 제한된 residual", color=INK, fontsize=13, loc="left")
    save(fig, "pp_architecture.png")


def fig_pp_equation_panel():
    """Single equation + term explanations."""
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.4, 3.15), 10.6, 1.1, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=WASH, edgecolor=TEAL, linewidth=2))
    ax.text(5.7, 3.7, r"$\hat{y}=\hat{y}_{\mathrm{affine}}+w(d)\,B\tanh(r_\theta(x)/B)$",
            ha="center", va="center", color=INK, fontsize=18)

    terms = [
        (0.4, r"$\hat{y}_{affine}$", "관측 밖에서도\n폭주하지 않는\n기본 추세", ORANGE),
        (3.15, r"$r_\theta(x)$", "history에서\n학습한\n비선형 보정", TEAL),
        (5.9, r"$B\tanh(\cdot/B)$", "보정 크기\n상한으로\n제한", NAVY),
        (8.65, r"$w(d)$", "support에서\n멀수록\n보정 감쇠", CRIMSON),
    ]
    for x, title, body, col in terms:
        ax.add_patch(FancyBboxPatch((x, 0.45), 2.5, 2.4, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor="white", edgecolor=col, linewidth=2.2))
        ax.add_patch(Rectangle((x, 2.45), 2.5, 0.4, facecolor=col, edgecolor="none"))
        ax.text(x + 1.25, 2.65, title, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.text(x + 1.25, 1.35, body, ha="center", va="center", color=INK, fontsize=12)
    ax.set_title("수식은 하나면 충분 — 각 항이 무엇을 제한하는가", color=INK, fontsize=13, loc="left")
    save(fig, "pp_equation_panel.png")


def fig_pp_shared_vs_data():
    """Shared extrapolation prior vs dataset-specific learning."""
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    rows = [
        (3.55, "공용 외삽 prior", "affine 기본 경로\nbounded residual\nsupport-distance gate", TEAL,
         "모든 데이터셋에\n같은 외삽 구조"),
        (2.05, "데이터별 학습", "affine 계수\nresidual NN\n보정 범위 · gate 강도", ORANGE,
         "추세와 residual만\n새로 맞춤"),
        (0.55, "도메인별 입력 표현", "배터리: capacity·rate·cycle\n기계: 센서 health·time history", NAVY,
         "식은 주입하지 않음\n관측 표현만 다름"),
    ]
    ax.add_patch(Rectangle((0.35, 0.35), 10.7, 4.15, facecolor=PALE, edgecolor=LINE, linewidth=1))
    for y, title, body, col, note in rows:
        ax.add_patch(FancyBboxPatch((0.55, y), 2.6, 1.25, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=col, edgecolor="none"))
        ax.text(1.85, y + 0.62, title, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.text(5.5, y + 0.62, body, ha="center", va="center", color=INK, fontsize=12)
        ax.text(9.5, y + 0.62, note, ha="center", va="center", color=GREY, fontsize=11)
    ax.set_title("무엇이 공용이고, 무엇이 데이터별인가", color=INK, fontsize=13, loc="left")
    save(fig, "pp_shared_vs_data.png")


def fig_support_adaptive():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    d = np.linspace(0, 4, 200)
    # weight toward local
    w = 1 - (1 / (1 + np.exp(-(d - 1.2) / 0.35))) * (1 - 0.4)
    bound = w * 2 + (1 - w) * 6
    ax.plot(d, bound, color=TEAL, lw=3.2, label="허용 residual 상한")
    ax.axvline(1.2, color=ORANGE, ls="--", lw=1.8)
    ax.text(1.28, 5.55, "전환점 τ", color=ORANGE, fontsize=11, fontweight="bold")
    ax.fill_between(d, 0, bound, where=d <= 1.2, color=TEAL, alpha=0.15)
    ax.fill_between(d, 0, bound, where=d >= 1.2, color=ORANGE, alpha=0.15)
    ax.text(0.35, 2.6, "local\n$B_L=2$", color=TEAL, fontsize=12, fontweight="bold")
    ax.text(2.5, 4.7, "broad\n$B_H=6$", color=ORANGE, fontsize=12, fontweight="bold")
    ax.set_xlabel("support 거리 (정규화)", color=GREY)
    ax.set_ylabel("residual envelope", color=GREY)
    ax.set_ylim(0, 6.8)
    ax.set_xlim(0, 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color(LINE)
    ax.tick_params(colors=GREY)
    ax.set_title("Support-adaptive: 가까우면 세밀, 멀면 유한 포화", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "support_adaptive.png")


def fig_main_results():
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    datasets = ["Sunwoda", "RWTH", "MICH"]
    models = {
        "고정 경계형": [0.939, 0.878, 0.468],
        "보정 제한 없음": [0.718, 0.788, 0.759],
        "거리 기반 보정": [0.894, 0.800, 0.736],
        "최종 모델": [0.934, 0.842, 0.751],
    }
    colors = [GREY, BLUE, ORANGE, TEAL]
    x = np.arange(len(datasets))
    width = 0.18
    for i, (name, vals) in enumerate(models.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=colors[i], edgecolor="white")
        for xi, v in zip(x + (i - 1.5) * width, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("설명력 R² (5회 평균)", color=GREY)
    ax.set_ylim(0, 1.15)
    ax.axhline(0, color=LINE, lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax.set_title("구조 비교 — 최종 모델이 평균과 어려운 데이터를 같이 올림", color=INK, fontsize=12, loc="left")
    save(fig, "main_ablation_bars.png")


def fig_competitor():
    # Algorithm-wise comparison with full TabPFN coverage where available
    datasets = ["HUST", "Virkler", "NASA", "Sunwoda", "RWTH", "MATR19", "MATRb2", "NCMAPSS"]
    algos = ["PP", "V-REx", "GroupDRO", "Monotone", "LinRBF", "Engression", "GP", "TabPFN"]
    mat = np.array(
        [
            [0.958, 0.809, 0.934, 0.822, 0.710, 0.878, -0.320, 0.218],
            [0.888, 0.583, 0.554, 0.565, 0.805, 0.552, 0.539, 0.621],
            [0.584, 0.285, 0.286, 0.283, 0.550, 0.549, 0.438, -0.691],
            [0.865, -0.240, -0.295, -0.048, 0.838, 0.619, -1.598, -0.886],
            [0.743, 0.645, 0.602, -0.005, 0.385, 0.526, -0.474, -2.175],
            [0.466, 0.044, 0.272, 0.018, -2.639, -0.726, -2.461, 0.202],
            [0.862, 0.850, 0.777, 0.674, -0.781, 0.739, 0.213, 0.618],
            [0.937, 0.883, 0.880, 0.892, 0.819, 0.932, 0.804, 0.934],
        ],
        dtype=float,
    )
    show = np.clip(mat, -0.5, 1.0)

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    im = ax.imshow(show, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=1.0)
    ax.set_xticks(np.arange(len(algos)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(algos, fontsize=10)
    ax.set_yticklabels(datasets, fontsize=11)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(len(datasets)):
        for j in range(len(algos)):
            v = mat[i, j]
            txt = f"{v:.2f}" if v > -1.5 else f"{v:.1f}"
            bold = j == 0 or (v == np.nanmax(mat[i]) and v > 0)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=INK if abs(show[i, j]) < 0.85 else "white",
                fontsize=8.5,
                fontweight="bold" if bold else "normal",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("설명력 R²", color=GREY)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("잘 된 8곳 · 알고리즘별 R² (TabPFN 전체 반영)", color=INK, fontsize=13, loc="left", pad=18)
    fig.text(
        0.08,
        0.01,
        "TabPFN: HUST/Virkler/NASA·C-MAPSS계=동일예산 · Sunwoda/RWTH/MATRb2=고정 test·train≤1000 · MATR19 보조 · NCMAPSS 단일 seed",
        color=GREY,
        fontsize=8,
    )
    save(fig, "competitor_bars.png")

    # Grouped bars
    fig2, ax2 = plt.subplots(figsize=(11.2, 4.8))
    key = ["PP", "V-REx", "GroupDRO", "LinRBF", "Engression", "TabPFN"]
    key_idx = [algos.index(k) for k in key]
    colors = [TEAL, ORANGE, BLUE, NAVY, GREY, CRIMSON]
    x = np.arange(len(datasets))
    w = 0.13
    for ki, (name, j) in enumerate(zip(key, key_idx)):
        vals = mat[:, j]
        plot_vals = np.clip(vals, -0.25, None)
        ax2.bar(x + (ki - 2.5) * w, plot_vals, w, label=name, color=colors[ki], edgecolor="white")
    ax2.axhline(0, color=LINE, lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=10)
    ax2.set_ylabel("설명력 R²", color=GREY)
    ax2.set_ylim(-0.3, 1.15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, ncol=6, loc="upper center", fontsize=8)
    ax2.set_title("주요 알고리즘 막대 비교 (TabPFN 포함)", color=INK, fontsize=12, loc="left")
    save(fig2, "competitor_algo_bars.png")

    # Dedicated PP vs TabPFN
    fig3, ax3 = plt.subplots(figsize=(10.6, 4.6))
    pp = mat[:, 0]
    pfn = mat[:, -1]
    x = np.arange(len(datasets))
    w = 0.36
    ax3.bar(x - w / 2, pp, w, color=TEAL, label="PP", edgecolor="white")
    ax3.bar(x + w / 2, np.clip(pfn, -1.0, None), w, color=CRIMSON, label="TabPFN", edgecolor="white")
    for i, (a, b) in enumerate(zip(pp, pfn)):
        ax3.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8, color=INK)
        ypos = max(b, -1.0) + 0.02 if b > -1.0 else -0.92
        ax3.text(i + w / 2, ypos, f"{b:.2f}", ha="center", fontsize=8, color=CRIMSON)
    ax3.axhline(0, color=LINE, lw=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, fontsize=10)
    ax3.set_ylabel("설명력 R²", color=GREY)
    ax3.set_ylim(-1.1, 1.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(frameon=False, loc="upper right")
    ax3.set_title("PP vs TabPFN (8곳 전체)", color=INK, fontsize=13, loc="left")
    fig3.text(0.12, 0.02, "보조 비교: TabPFN은 train 행 상한·일부 단일 seed 조건이 포함됨", color=GREY, fontsize=9)
    save(fig3, "pp_vs_tabpfn.png")


def fig_fail_success_map():
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    # scatter style map
    points = [
        (0.9, 0.85, "HUST", TEAL),
        (0.75, 0.78, "Virkler", TEAL),
        (0.55, 0.55, "NASA", TEAL),
        (0.8, 0.7, "Sunwoda", TEAL),
        (0.7, 0.65, "RWTH", TEAL),
        (0.5, 0.45, "MATR2019", ORANGE),
        (0.78, 0.72, "MATR b2", TEAL),
        (0.88, 0.82, "N-CMAPSS", TEAL),
        (0.2, 0.15, "MICH*", RED),
        (0.18, 0.2, "XJTU", RED),
        (0.15, 0.12, "FEMTO", RED),
        (0.12, 0.08, "Milling", RED),
    ]
    ax.axhline(0.5, color=LINE, ls=":", lw=1)
    ax.axvline(0.5, color=LINE, ls=":", lw=1)
    ax.fill_between([0.5, 1], 0.5, 1, color=TEAL, alpha=0.08)
    ax.fill_between([0, 0.5], 0, 0.5, color=RED, alpha=0.08)
    for x, y, name, c in points:
        ax.scatter([x], [y], s=120, color=c, edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(x + 0.02, y + 0.03, name, color=c, fontsize=9, fontweight="bold")
    ax.text(0.72, 0.92, "성공 영역", color=TEAL, fontsize=12, fontweight="bold")
    ax.text(0.08, 0.42, "실패/기권 영역", color=RED, fontsize=12, fontweight="bold")
    ax.text(0.55, 0.08, "경계·보정 필요", color=ORANGE, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("적용 조건 적합도 (개념축)", color=GREY)
    ax.set_ylabel("외삽 성능 (개념축)", color=GREY)
    ax.set_title("실험 지형: 성공·경계·실패를 같은 지도에 둔다 (*기본 PP 기준)", color=INK, fontsize=12, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "experiment_map.png")


def fig_bootstrap():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    names = ["HUST", "RWTH", "MATR2019"]
    means = [-25.19, -23.27, -10.63]
    lo = [-34.38, -41.79, -15.67]
    hi = [-14.54, -8.94, -6.37]
    y = np.arange(len(names))
    ax.axvline(0, color=RED, lw=1.5, ls="--")
    for i, (m, l, h) in enumerate(zip(means, lo, hi)):
        ax.plot([l, h], [i, i], color=TEAL, lw=4, solid_capstyle="round")
        ax.scatter([m], [i], color=NAVY, s=70, zorder=3)
        ax.text(h + 1.2, i, f"{m:.1f}  [{l:.1f}, {h:.1f}]", va="center", color=GREY, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("unit RMSE 변화 (보정 후 − 전) · 95% bootstrap CI", color=GREY)
    ax.set_title("Unit-level paired bootstrap: CI가 0 아래 = 일관된 개선", color=INK, fontsize=12, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    save(fig, "bootstrap_ci.png")


def fig_pp_pae_flow():
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    stages = [
        (0.35, "분기", "외삽 문제",
         "후보식·관측 계약만 본다\n데이터셋 이름으로\n경로를 고르지 않음", GREY),
        (3.9, "식이 있을 때", "PAE",
         "식 = 뼈대\nNN = 부족분(제한 보정)\n실패 시 끔 / PP로", ORANGE),
        (7.45, "식이 없을 때", "PP",
         "equation-free safeguard\n계약상 약한 bundle만\n기계적으로 켠다", TEAL),
    ]
    for x, title, tag, body, col in stages:
        ax.add_patch(FancyBboxPatch((x, 1.35), 3.1, 2.7, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    facecolor="white", edgecolor=col, linewidth=2.5))
        ax.add_patch(Rectangle((x, 3.7), 3.1, 0.35, facecolor=col, edgecolor="none"))
        ax.text(x + 1.55, 3.87, title, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.text(x + 1.55, 3.25, tag, ha="center", color=col, fontsize=12, fontweight="bold")
        ax.text(x + 1.55, 2.15, body, ha="center", va="center", color=INK, fontsize=11)
    for x in [3.45, 7.0]:
        ax.annotate("", xy=(x + 0.35, 2.7), xytext=(x, 2.7),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=2.2))

    ax.add_patch(FancyBboxPatch((0.35, 0.25), 10.5, 0.85, boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor=WASH, edgecolor=LINE))
    ax.text(5.6, 0.68, "PAE도 식만으로 끝내지 않는다  ·  허용된 식 + 제한된 NN 보정",
            ha="center", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_title("문제 계약 → 후보식 유무로 분기 → PAE(식+NN) / PP(식 없음)", color=INK, fontsize=13, loc="left")
    save(fig, "pp_pae_flow.png")


def fig_pae_equation_nn():
    """PAE: verified equation + bounded NN covering the gap."""
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # soft wash band
    ax.add_patch(FancyBboxPatch((0.25, 0.35), 10.9, 4.5, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=PALE, edgecolor=LINE, linewidth=1.2))

    # pipeline nodes
    nodes = [
        (0.45, 2.55, 1.85, 1.7, "① 후보식", "문헌 · LLM\n물리·경험식", ORANGE),
        (2.55, 2.55, 1.85, 1.7, "② 검증", "변수·단위\ntrain-only", CRIMSON),
        (4.65, 2.55, 2.55, 1.7, "③ 조립", "식 뼈대 + NN 부족분\n보정은 제한", NAVY),
        (7.5, 2.55, 1.7, 1.7, "④ 판정", "이득 있음?\non / off", TEAL),
        (9.5, 2.55, 1.55, 1.7, "⑤ 출력", "예측\n또는 거절", BLUE),
    ]
    for x, y, w, h, title, body, col in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor="white", edgecolor=col, linewidth=2.2))
        ax.add_patch(Rectangle((x, y + h - 0.38), w, 0.38, facecolor=col, edgecolor="none"))
        ax.text(x + w / 2, y + h - 0.19, title, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + 0.65, body, ha="center", va="center", color=INK, fontsize=11)
    for x in [2.3, 4.4, 7.2, 9.2]:
        ax.annotate("", xy=(x + 0.2, 3.4), xytext=(x, 3.4),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=2))

    # equation + NN split panel
    ax.add_patch(FancyBboxPatch((0.55, 0.55), 4.7, 1.7, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="white", edgecolor=ORANGE, linewidth=2))
    ax.text(2.9, 1.95, "식 (뼈대)", ha="center", color=ORANGE, fontsize=12, fontweight="bold")
    ax.text(2.9, 1.25, "방향 · 경계 · 관계\n검증된 항만 고정", ha="center", color=INK, fontsize=11)

    ax.add_patch(FancyBboxPatch((5.55, 0.55), 5.2, 1.7, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="white", edgecolor=TEAL, linewidth=2))
    ax.text(8.15, 1.95, "NN (부족분)", ha="center", color=TEAL, fontsize=12, fontweight="bold")
    ax.text(8.15, 1.25, "미지 파라미터 · 잔차 · 노이즈\n식 위에 제한적으로 학습", ha="center", color=INK, fontsize=11)

    ax.annotate("", xy=(5.55, 1.4), xytext=(5.25, 1.4),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.5))
    ax.text(5.4, 1.7, "+", ha="center", color=NAVY, fontsize=16, fontweight="bold")

    ax.set_title("PAE = 허용된 식 + 제한된 NN 보정  (식이 있어도 부족분은 학습으로 메움)",
                 color=INK, fontsize=13, loc="left", pad=4)
    save(fig, "pae_equation_nn.png")


def fig_pp_vs_pae_split():
    """Side-by-side role visual for deck."""
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # left PP
    ax.add_patch(FancyBboxPatch((0.3, 0.4), 5.2, 3.8, boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor="white", edgecolor=TEAL, linewidth=2.8))
    ax.add_patch(Rectangle((0.3, 3.7), 5.2, 0.5, facecolor=TEAL, edgecolor="none"))
    ax.text(2.9, 3.95, "PP  ·  equation-free safeguard", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
    ax.text(2.9, 3.15, "명시적 도메인 식 없음", ha="center", color=INK, fontsize=13, fontweight="bold")
    for i, t in enumerate([
        "일반 구조 bias만 (affine · 거리감쇠 · 거절)",
        "계약 → 고정 rule로 weak bundle 기계적 활성",
        "중심 질문: 식 없이 외삽 형태를 어떻게 통제할까",
    ]):
        ax.text(0.65, 2.45 - i * 0.55, "·  " + t, ha="left", color=GREY, fontsize=11)

    # right PAE
    ax.add_patch(FancyBboxPatch((5.9, 0.4), 5.2, 3.8, boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor="white", edgecolor=ORANGE, linewidth=2.8))
    ax.add_patch(Rectangle((5.9, 3.7), 5.2, 0.5, facecolor=ORANGE, edgecolor="none"))
    ax.text(8.5, 3.95, "PAE  ·  equation prior + NN", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
    ax.text(8.5, 3.15, "후보식 검증 후 식+NN 조립", ha="center", color=INK, fontsize=13, fontweight="bold")
    for i, t in enumerate([
        "식 = 뼈대, NN = 부족분(제한 보정)",
        "이득 없으면 식 끔 · PP/거절로 fallback",
        "중심 질문: 어떤 식을 어떻게 넣을까",
    ]):
        ax.text(6.25, 2.45 - i * 0.55, "·  " + t, ha="left", color=GREY, fontsize=11)

    ax.set_title("두 논문의 경계 — 같은 ‘외삽’이어도 입력이 다름", color=INK, fontsize=13, loc="left")
    save(fig, "pp_vs_pae_split.png")


def fig_dataset_split():
    """Common PP split recipe for appendix."""
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    steps = [
        (0.3, "① 계약 고정", "경계·unit·시간축\n인과 입력·외삽 좌표", ORANGE),
        (2.55, "② unit 분리", "같은 개체가\ntrain/val/test에\n동시에 안 들어감", TEAL),
        (4.8, "③ train support", "학습 unit의\n관측 범위 안만 사용", BLUE),
        (7.05, "④ hull-out만", "val/test는\n사전 좌표가\ntrain hull 밖인 행", CRIMSON),
        (9.3, "⑤ 선택·보고", "val만으로 설정\ntest R² 한 번 개봉", NAVY),
    ]
    for x, t, b, c in steps:
        ax.add_patch(FancyBboxPatch((x, 1.35), 1.95, 2.7, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor="white", edgecolor=c, linewidth=2.2))
        ax.add_patch(Rectangle((x, 3.7), 1.95, 0.35, facecolor=c, edgecolor="none"))
        ax.text(x + 0.975, 3.87, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + 0.975, 2.4, b, ha="center", va="center", color=INK, fontsize=10)
        if x < 9:
            ax.annotate("", xy=(x + 2.15, 2.7), xytext=(x + 1.95, 2.7),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8))

    ax.add_patch(FancyBboxPatch((0.3, 0.25), 10.8, 0.85, boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor=WASH, edgecolor=LINE))
    ax.text(5.7, 0.68, "스케일·정규화는 train만  ·  seed 42–46  ·  주 지표: raw pooled R²",
            ha="center", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_title("공통 스플릿: 개체 분리 + 학습 support 밖(끝단)만 평가", color=INK, fontsize=13, loc="left")
    save(fig, "dataset_split.png")


def fig_dataset_landscape():
    """Where PP works vs fails across dataset types."""
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    blocks = [
        (0.3, 2.55, 5.3, 2.0, "잘 되는 축", TEAL,
         "열화 좌표 끝단 · 처음 보는 unit\n경계·방향이 맞고 val/test 이동이 비슷\nHUST · Virkler · NASA · Sunwoda\nRWTH · MATR · N-CMAPSS"),
        (5.9, 2.55, 5.2, 2.0, "안 되는 / 거절이 맞음", CRIMSON,
         "관계·메커니즘이 바뀜\n끝점 1개·유닛 너무 적음\nXJTU · FEMTO · milling\n기본 MICH → 경계형으로만 일부 회복"),
        (0.3, 0.35, 10.8, 1.9, "추론 대상 (공통)", NAVY,
         "주 타깃 = 남은수명(RUL) 또는 동등한 잔여량\n"
         "입력 = 인과적으로 관측 가능한 건강/부하/이력 요약 (미래·test 라벨 통계 금지)\n"
         "판정 = pooled R² + unit coverage + seed 안정성 + 경계 위반 0"),
    ]
    for x, y, w, h, title, col, body in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
                                    facecolor="white", edgecolor=col, linewidth=2.4))
        ax.add_patch(Rectangle((x, y + h - 0.42), w, 0.42, facecolor=col, edgecolor="none"))
        ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center", color=INK, fontsize=11)
    ax.set_title("데이터셋 지도 — 무엇을 맞히고, 언제 점수가 의미 있나", color=INK, fontsize=13, loc="left")
    save(fig, "dataset_landscape.png")


def fig_mich_units():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    units = [25, 26, 27, 28, 29, 30, 31, 32]
    r2 = [0.789, 0.609, 0.813, 0.761, 0.657, 0.722, 0.496, 0.957]
    colors = [TEAL if v >= 0.6 else ORANGE for v in r2]
    bars = ax.bar([str(u) for u in units], r2, color=colors, edgecolor="white")
    ax.axhline(0, color=LINE, lw=1)
    ax.axhline(0.751, color=NAVY, ls="--", lw=1.5, label="pooled 0.751")
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("within-unit R²", color=GREY)
    ax.set_xlabel("MICH test unit", color=GREY)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("MICH: 8/8 unit 양의 R² (unit 31 복구 0.496)", color=INK, fontsize=12, loc="left")
    save(fig, "mich_units.png")


def fig_protocol():
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    steps = [
        (0.3, "① 나누기", "다른 개체끼리\n진짜 바깥만 테스트", ORANGE),
        (2.9, "② 고르기", "검증 구간만으로\n설정 결정", TEAL),
        (5.5, "③ 학습", "여러 번 재학습\n기본 경향은 고정", BLUE),
        (8.1, "④ 보고", "점수·안정성\n경계 위반 여부", NAVY),
    ]
    for x, t, b, c in steps:
        ax.add_patch(FancyBboxPatch((x, 0.7), 2.3, 2.2, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor="white", edgecolor=c, lw=2))
        ax.text(x + 1.15, 2.4, t, ha="center", color=c, fontsize=14, fontweight="bold")
        ax.text(x + 1.15, 1.5, b, ha="center", color=INK, fontsize=12)
        if x < 8:
            ax.annotate("", xy=(x + 2.55, 1.8), xytext=(x + 2.3, 1.8),
                        arrowprops=dict(arrowstyle="->", color=LINE, lw=2))
    ax.text(5.4, 0.25, "테스트 정답은 모델 고를 때 보지 않는다", ha="center", color=CRIMSON, fontsize=11, fontweight="bold")
    ax.set_title("검증 흐름", color=INK, fontsize=13, loc="left")
    save(fig, "protocol_flow.png")


def main():
    fig_motivation()
    fig_prior_ladder()
    fig_pp_core_idea()
    fig_pp_architecture()
    fig_pp_equation_panel()
    fig_pp_shared_vs_data()
    fig_support_adaptive()
    fig_main_results()
    fig_competitor()
    fig_fail_success_map()
    fig_bootstrap()
    fig_pp_pae_flow()
    fig_pae_equation_nn()
    fig_pp_vs_pae_split()
    fig_dataset_split()
    fig_dataset_landscape()
    fig_mich_units()
    fig_protocol()
    print("done", OUT)


if __name__ == "__main__":
    main()
