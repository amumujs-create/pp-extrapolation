#!/usr/bin/env python3
"""PP-X vs Engression win/loss descriptors (train/val only).

Pre-declared descriptors (no test Y):
  1. unit_heterogeneity — same-health-bin cross-unit RUL dispersion
  2. conditional_dispersion — local Y spread among X-neighbors
  3. boundary_informativeness — R² of Y ~ failure-boundary distance
  4. regime_heterogeneity — slope(Y~health) variation across regimes

Correlate each with ΔR² = R²_PP-X − R²_Engression from equal-budget summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"),
                str(ROOT.parent / "ca-css-ncmapss")]

from cross_domain_mechanism_study import all_datasets  # noqa: E402

OUT = ROOT / "results" / "ppx_engression_winloss_descriptors_v1"
SUMMARY = ROOT / "results" / "full_equal_candidate_budget_summary_v1" / "results.json"
DOC = ROOT / "PPX_ENGRESSION_WINLOSS_DESCRIPTORS_KO.md"

# Explicit failure boundaries used by PP-X contracts where defined.
# direction: health decreases toward boundary, or crack increases toward it.
BOUNDARY = {
    "hust": ("decreasing", 0.880),
    "virkler": ("increasing", 49.8),
    "nasa_battery": ("decreasing", 0.0),  # health_phi → 0
    "sunwoda": ("decreasing", 880.0),
    "rwth": ("decreasing", 0.8),
    "mich": ("decreasing", 0.8),  # relative capacity convention; may be weak
    "matr": ("decreasing", None),  # filled from train coordinate min
    "matr_batch2": ("decreasing", None),
    "ncmapss": None,  # no scalar capacity/crack EOL
}

NAME_MAP = {
    "HUST": "hust",
    "Virkler": "virkler",
    "NASA": "nasa_battery",
    "SUNWODA": "sunwoda",
    "RWTH": "rwth",
    "MICH": "mich",
    "MATR2019": "matr",
    "MATR-b2": "matr_batch2",
    "N-CMAPSS": "ncmapss",
}


def _r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    den = float(np.sum((y - y.mean()) ** 2))
    if den <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / den)


def _pool(tr, va):
    keys = ("x", "y", "groups")
    out = {k: np.concatenate([np.asarray(tr[k]), np.asarray(va[k])]) for k in keys}
    if "coordinate" in tr and "coordinate" in va:
        out["coordinate"] = np.concatenate(
            [np.asarray(tr["coordinate"]), np.asarray(va["coordinate"])]
        )
    return out


def _boundary_distance(x0, direction, boundary):
    x0 = np.asarray(x0, float)
    if direction == "decreasing":
        return x0 - float(boundary)
    if direction == "increasing":
        return float(boundary) - x0
    raise ValueError(direction)


def unit_heterogeneity(x0, y, g, n_bins=8):
    """Cross-unit RUL std within health quantile bins / global Y std."""
    x0 = np.asarray(x0, float)
    y = np.asarray(y, float)
    g = np.asarray(g)
    sy = float(np.std(y))
    if sy < 1e-12 or len(y) < 20:
        return float("nan"), {"n_bins_used": 0}
    qs = np.unique(np.quantile(x0, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return float("nan"), {"n_bins_used": 0}
    vals = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x0 >= lo) & (x0 <= hi if hi == qs[-1] else x0 < hi)
        units = np.unique(g[m])
        if len(units) < 2:
            continue
        means = [float(np.mean(y[m & (g == u)])) for u in units]
        vals.append(float(np.std(means)))
    if not vals:
        return float("nan"), {"n_bins_used": 0}
    return float(np.median(vals) / sy), {"n_bins_used": len(vals), "raw_median_std": float(np.median(vals))}


def conditional_dispersion(x, y, k=25):
    """Median neighbor Y-std / global Y-std in standardized X space."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(y)
    sy = float(np.std(y))
    if sy < 1e-12 or n < k + 5:
        return float("nan"), {"k": k, "n": n}
    mu = x.mean(0)
    sd = np.maximum(x.std(0), 1e-8)
    z = (x - mu) / sd
    # subsample anchors for large n
    rng = np.random.default_rng(20260912)
    anchors = np.arange(n) if n <= 2500 else rng.choice(n, 2500, replace=False)
    local = []
    for i in anchors:
        d2 = np.sum((z - z[i]) ** 2, axis=1)
        nn = np.argpartition(d2, k)[: k + 1]
        nn = nn[nn != i][:k]
        if len(nn) < max(5, k // 2):
            continue
        local.append(float(np.std(y[nn])))
    if not local:
        return float("nan"), {"k": k, "n": n}
    return float(np.median(local) / sy), {"k": k, "n_anchors": len(local)}


def boundary_informativeness(x0, y, boundary_spec, coordinate=None):
    """Train/val R² of linear Y ~ failure-boundary distance."""
    x0 = np.asarray(x0, float)
    y = np.asarray(y, float)
    if boundary_spec is None:
        # No scalar EOL: report R² of Y ~ x0 (gate coordinate) as weak proxy,
        # and flag has_explicit_boundary=False.
        a = np.column_stack([np.ones(len(x0)), x0])
        coef = np.linalg.lstsq(a, y, rcond=None)[0]
        return _r2(y, a @ coef), {
            "has_explicit_boundary": False,
            "proxy": "linear_Y_on_x0",
            "boundary": None,
        }
    direction, boundary = boundary_spec
    if boundary is None:
        if coordinate is None:
            boundary = float(np.min(x0))
        else:
            boundary = float(np.min(np.asarray(coordinate, float)))
    dist = _boundary_distance(x0, direction, boundary)
    a = np.column_stack([np.ones(len(dist)), dist])
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return _r2(y, a @ coef), {
        "has_explicit_boundary": True,
        "direction": direction,
        "boundary": float(boundary),
        "coef_slope": float(coef[1]),
    }


def _regimes_for(name, tr, va, pooled):
    g = pooled["groups"]
    x0 = pooled["x"][:, 0]
    if name == "hust":
        # HUST_p-c → protocol p
        labs = np.array([str(u).split("_")[-1].split("-")[0] for u in g])
        return labs
    if name == "sunwoda":
        return (np.asarray(g).astype(int) % 1000).astype(str)
    if name == "ncmapss":
        # TRA tertiles as operating-regime proxy
        qs = np.quantile(x0, [0, 1 / 3, 2 / 3, 1])
        labs = np.digitize(x0, qs[1:-1], right=False).astype(str)
        return labs
    if name in ("matr", "matr_batch2"):
        # charging/cell cohort proxy: unit id tertile by mean health
        units = np.unique(g)
        means = {u: float(np.mean(x0[g == u])) for u in units}
        vals = np.array(list(means.values()))
        cuts = np.quantile(vals, [1 / 3, 2 / 3])
        unit_lab = {}
        for u, m in means.items():
            unit_lab[u] = "0" if m < cuts[0] else ("1" if m < cuts[1] else "2")
        return np.array([unit_lab[u] for u in g])
    # Default: 3 k-means-like bins on x0 (quantile regimes)
    qs = np.quantile(x0, [0, 1 / 3, 2 / 3, 1])
    return np.digitize(x0, qs[1:-1], right=False).astype(str)


def regime_heterogeneity(x0, y, regimes):
    """MAD of per-regime slopes of Y~health, scaled by |global slope|."""
    x0 = np.asarray(x0, float)
    y = np.asarray(y, float)
    regimes = np.asarray(regimes)
    slopes = []
    for r in np.unique(regimes):
        m = regimes == r
        if m.sum() < 30:
            continue
        a = np.column_stack([np.ones(m.sum()), x0[m]])
        b = np.linalg.lstsq(a, y[m], rcond=None)[0]
        slopes.append(float(b[1]))
    if len(slopes) < 2:
        return float("nan"), {"n_regimes": len(slopes)}
    a_all = np.column_stack([np.ones(len(x0)), x0])
    g_slope = float(np.linalg.lstsq(a_all, y, rcond=None)[0][1])
    mad = float(np.median(np.abs(np.asarray(slopes) - np.median(slopes))))
    scale = max(abs(g_slope), 1e-8)
    return float(mad / scale), {
        "n_regimes": len(slopes),
        "slopes": slopes,
        "global_slope": g_slope,
    }


def descriptors_for_split(name, tr, va):
    pooled = _pool(tr, va)
    x = pooled["x"]
    y = pooled["y"]
    g = pooled["groups"]
    x0 = x[:, 0]
    coord = pooled.get("coordinate")

    uh, uh_m = unit_heterogeneity(x0, y, g)
    cd, cd_m = conditional_dispersion(x, y)
    bi, bi_m = boundary_informativeness(x0, y, BOUNDARY.get(name), coordinate=coord)
    regs = _regimes_for(name, tr, va, pooled)
    rh, rh_m = regime_heterogeneity(x0, y, regs)

    return {
        "unit_heterogeneity": uh,
        "conditional_dispersion": cd,
        "boundary_informativeness": bi,
        "regime_heterogeneity": rh,
        "meta": {
            "n_train": int(len(tr["y"])),
            "n_val": int(len(va["y"])),
            "n_pooled": int(len(y)),
            "n_units": int(len(np.unique(g))),
            "unit_heterogeneity": uh_m,
            "conditional_dispersion": cd_m,
            "boundary_informativeness": bi_m,
            "regime_heterogeneity": rh_m,
        },
    }


def descriptors_dataset(name, payload):
    if name == "nasa_battery":
        rows = []
        for fold in payload["folds"]:
            rows.append(descriptors_for_split(name, fold["train"], fold["validation"]))
        keys = [
            "unit_heterogeneity",
            "conditional_dispersion",
            "boundary_informativeness",
            "regime_heterogeneity",
        ]
        out = {k: float(np.nanmedian([r[k] for r in rows])) for k in keys}
        out["meta"] = {"folds": [r["meta"] for r in rows], "aggregate": "nanmedian"}
        return out
    tr, va, _te = payload
    return descriptors_for_split(name, tr, va)


def load_delta_r2():
    summary = json.loads(SUMMARY.read_text())
    rows = {}
    for row in summary["datasets"]:
        key = NAME_MAP[row["dataset"]]
        ppx = float(row["ppx_r2"])
        eng = float(row["baseline_r2"]["engression"])
        rows[key] = {
            "display": row["dataset"],
            "ppx_r2": ppx,
            "engression_r2": eng,
            "delta_r2": ppx - eng,
        }
    return rows


def correlate(table, y_key="delta_r2"):
    keys = [
        "unit_heterogeneity",
        "conditional_dispersion",
        "boundary_informativeness",
        "regime_heterogeneity",
    ]
    y = np.array([r[y_key] for r in table], float)
    out = {}
    for k in keys:
        x = np.array([r[k] for r in table], float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            out[k] = {"n": int(m.sum()), "spearman_rho": None, "p": None}
            continue
        rho, p = spearmanr(x[m], y[m])
        if y_key == "delta_r2":
            hyp = (
                "negative_with_delta_r2"
                if k in ("unit_heterogeneity", "conditional_dispersion", "regime_heterogeneity")
                else "positive_with_delta_r2"
            )
        elif y_key == "engression_r2":
            hyp = (
                "positive_with_engression_r2"
                if k in ("unit_heterogeneity", "conditional_dispersion", "regime_heterogeneity")
                else "negative_or_weak_with_engression_r2"
            )
        else:
            hyp = "exploratory"
        out[k] = {
            "n": int(m.sum()),
            "spearman_rho": float(rho),
            "p": float(p),
            "hypothesis": hyp,
        }
    return out


def write_doc(payload):
    lines = [
        "# PP-X vs Engression 승패 설명변수 (v1)",
        "",
        "> Equal-budget ΔR² = R²_PP-X − R²_Engression 과 train/val-only 네 설명변수의",
        "> Spearman 상관. test Y는 설명변수 계산에 사용하지 않았다.",
        "",
        "## 판정 요약",
        "",
        "단순 가설(`conditional_dispersion↑ ⇒ ΔR²↓`, `boundary_informativeness↑ ⇒ ΔR²↑`)은",
        "**전 9-setting 순위상관으로는 확증되지 않았다.** 다만 패턴은 더 세분화된다.",
        "",
        "### vs ΔR² (전체 9)",
        "",
    ]
    for k, v in payload["correlations"]["vs_delta_r2"].items():
        rho = v.get("spearman_rho")
        p = v.get("p")
        if rho is None:
            lines.append(f"- `{k}`: 유효 표본 부족")
        else:
            lines.append(
                f"- `{k}`: Spearman ρ={rho:.3f}, p={p:.4f} "
                f"(가설: {v['hypothesis']}, n={v['n']})"
            )
    lines += [
        "",
        "### vs Engression R² (전체 9)",
        "",
    ]
    for k, v in payload["correlations"]["vs_engression_r2"].items():
        rho = v.get("spearman_rho")
        p = v.get("p")
        if rho is None:
            lines.append(f"- `{k}`: 유효 표본 부족")
        else:
            lines.append(f"- `{k}`: ρ={rho:.3f}, p={p:.4f}, n={v['n']}")
    lines += [
        "",
        "### 민감도: Engression R²≥0 인 7-setting만 (MICH·MATR2019 제외)",
        "",
    ]
    for k, v in payload["correlations"]["vs_delta_r2_eng_nonneg"].items():
        rho = v.get("spearman_rho")
        p = v.get("p")
        if rho is None:
            lines.append(f"- `{k}`: 유효 표본 부족")
        else:
            lines.append(f"- `{k}`: ρ={rho:.3f}, p={p:.4f}, n={v['n']}")
    lines += [
        "",
        "## 설정별 표",
        "",
        "| Setting | PP-X R² | Engression R² | ΔR² | unit_het | cond_disp | bound_info | regime_het |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["rows"]:
        rh = r["regime_heterogeneity"]
        rh_s = "nan" if rh is None or not np.isfinite(rh) else f"{rh:.3f}"
        lines.append(
            f"| {r['display']} | {r['ppx_r2']:.3f} | {r['engression_r2']:.3f} | "
            f"{r['delta_r2']:+.3f} | {r['unit_heterogeneity']:.3f} | "
            f"{r['conditional_dispersion']:.3f} | {r['boundary_informativeness']:.3f} | "
            f"{rh_s} |"
        )
    lines += [
        "",
        "## 정의 (사전 고정)",
        "",
        "1. **unit_heterogeneity**: health 분위 구간 안에서 unit-mean RUL의 표준편차 중앙값 / 전역 std(Y).",
        "2. **conditional_dispersion**: 표준화 X kNN(k=25)의 이웃 Y 표준편차 중앙값 / 전역 std(Y).",
        "3. **boundary_informativeness**: 명시적 failure boundary 거리로 Y를 선형 설명한 R². "
        "N-CMAPSS는 스칼라 EOL이 없어 Y~x0(TRA) 대용이며 `has_explicit_boundary=false`.",
        "4. **regime_heterogeneity**: regime별 Y~health 기울기의 MAD / |전역 기울기|. "
        "HUST=protocol, Sunwoda=온도, N-CMAPSS=TRA 삼분위, 그 외 health 삼분위/코호트.",
        "",
        "## 해석",
        "",
        "1. **N-CMAPSS 단독 패턴**은 가설과 맞다: conditional_dispersion이 9개 중 최대(0.82), "
        "명시적 boundary informativeness≈0, ΔR²≈0(+0.005)로 Engression이 사실상 동률.",
        "2. **전역 Spearman**에서는 unit_heterogeneity가 ΔR²와 **양의** 상관(ρ≈0.77). "
        "즉 ‘이질성↑ → Engression 유리’가 아니라, 같은 health 근처 unit RUL이 흩어질수록 "
        "Engression이 크게 무너지고 PP-X 이득(ΔR²)이 커지는 쪽에 가깝다 "
        "(MICH·MATR2019 Engression 음수 R²가 순위을 당김).",
        "3. boundary_informativeness의 전역 상관은 가설 반대 부호·비유의. "
        "Sunwoda·Virkler처럼 boundary R²가 높아도 ΔR² 크기는 제각각이고, "
        "MATR-b2는 boundary R²가 낮은데도 Engression이 양수라 ΔR²가 작다.",
        "4. Engression R²≥0 인 7개만 보면 상관 부호·크기가 다시 흔들린다 → "
        "**n=9 정적 설명변수만으로 승패 법칙을 확정할 수 없다.**",
        "5. 논문용 문장 후보: “분포적 이질성이 큰 N-CMAPSS에서는 Engression이 경쟁적이지만, "
        "setting-level 정적 descriptor만으로 PP-X−Engression 이득을 보편 예측하는 사전 gate는 "
        "아직 성립하지 않는다. 특히 unit-level health 근처 RUL 이질성은 Engression 붕괴와 "
        "함께 PP-X 상대이득을 키우는 방향으로 관측됐다.”",
        "",
        "## 해석 경계",
        "",
        "- n=9 setting이라 상관계수는 **탐색적**이다. p-value를 확증으로 쓰지 않는다.",
        "- Sunwoda train/val이 동일 온도(25°C)라 regime_heterogeneity는 NaN.",
        "- N-CMAPSS regime_het는 TRA 삼분위 기울기 차이로 매우 큼(≈9) — 스케일 이상치.",
        "",
        f"- Artifact: `{OUT.relative_to(ROOT)}/results.json`",
        f"- Runner: `experiments/ppx_engression_winloss_descriptors.py`",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    deltas = load_delta_r2()
    data = all_datasets()
    rows = []
    details = {}
    order = list(NAME_MAP.values())
    rev = {v: k for k, v in NAME_MAP.items()}
    for name in order:
        print("DESC", name, flush=True)
        d = descriptors_dataset(name, data[name])
        details[name] = d
        row = {
            "dataset": name,
            "display": rev[name],
            **deltas[name],
            "unit_heterogeneity": d["unit_heterogeneity"],
            "conditional_dispersion": d["conditional_dispersion"],
            "boundary_informativeness": d["boundary_informativeness"],
            "regime_heterogeneity": d["regime_heterogeneity"],
        }
        rows.append(row)
        rh = row["regime_heterogeneity"]
        rh_s = "nan" if rh is None or not np.isfinite(rh) else f"{rh:.3f}"
        print(
            f"  dR2={row['delta_r2']:+.3f} uh={row['unit_heterogeneity']:.3f} "
            f"cd={row['conditional_dispersion']:.3f} bi={row['boundary_informativeness']:.3f} "
            f"rh={rh_s}",
            flush=True,
        )

    nonneg = [r for r in rows if r["engression_r2"] >= 0]
    corr = {
        "vs_delta_r2": correlate(rows, "delta_r2"),
        "vs_engression_r2": correlate(rows, "engression_r2"),
        "vs_delta_r2_eng_nonneg": correlate(nonneg, "delta_r2"),
    }
    loo = []
    for i in range(len(rows)):
        sub = [r for j, r in enumerate(rows) if j != i]
        loo.append({"drop": rows[i]["dataset"], "correlations": correlate(sub, "delta_r2")})

    payload = {
        "protocol": (
            "train+validation only descriptors; equal-budget PP-X vs Engression "
            "ΔR² from full_equal_candidate_budget_summary_v1; no test Y in descriptors"
        ),
        "rows": rows,
        "correlations": corr,
        "leave_one_setting_out": loo,
        "details": details,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_doc(payload)
    print(json.dumps(corr, indent=2))
    print("WROTE", OUT / "results.json")
    print("WROTE", DOC)


if __name__ == "__main__":
    main()
