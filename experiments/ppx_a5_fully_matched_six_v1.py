#!/usr/bin/env python3
"""Combine Battery-3 and non-battery matched A5 controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from ppx_final_ablation_statistics import compare  # noqa: E402

OUT = ROOT / "results/ppx_a5_fully_matched_six_v1"
REPORT = ROOT / "PPX_A5_FULLY_MATCHED_SIX_V1_KO.md"
RNG = np.random.default_rng(20260917)


def settings():
    rows = []
    b = np.load(ROOT / "results/bq_pp_matched_controls_v1/predictions.npz", allow_pickle=True)
    for code, name in enumerate(("Sunwoda", "RWTH", "MICH")):
        take = b["dataset"] == code
        # Use the unbounded BQ residual here: bounded/dual executors would
        # change correction capacity and contaminate the pure prior contrast.
        rows.append((name, b["y"][take], b["units"][take], b["direct_nn"][:, take], b["frozen_unbounded_pp"][:, take]))
    for name, stem in (("HUST", "hust"), ("MATR-b2", "matr_b2"), ("N-CMAPSS", "n_cmapss")):
        z = np.load(ROOT / f"results/ppx_a5_fully_matched_nonbattery_v1/{stem}.npz", allow_pickle=True)
        rows.append((name, z["test_y"], z["test_groups"], z["direct_test"], z["prior_test"]))
    return rows


def bh(rows):
    p = np.asarray([row["unit_signflip_p_two_sided"] for row in rows])
    order = np.argsort(p)
    q = np.empty(len(p)); running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        index = order[rank]
        running = min(running, p[index] * len(p) / (rank + 1))
        q[index] = running
    for row, value in zip(rows, q):
        row["unit_signflip_q_bh"] = float(min(value, 1.0))


def main():
    rows = []
    for name, y, groups, direct, prior in settings():
        result = compare(name, "A5 no-prior NN vs prior+same residual", y, groups, direct, prior, status="fully_matched")
        direct_rmse = float(np.sqrt(np.mean((y - direct.mean(0)) ** 2)))
        rows.append({
            "setting": name,
            "direct_r2": float(r2_score(y, direct.mean(0))),
            "prior_residual_r2": float(r2_score(y, prior.mean(0))),
            "delta_r2": float(r2_score(y, prior.mean(0)) - r2_score(y, direct.mean(0))),
            "mean_unit_rmse_reduction": result["mean_unit_rmse_reduction"],
            "normalized_unit_rmse_reduction": float(result["mean_unit_rmse_reduction"] / max(direct_rmse, 1e-12)),
            "unit_bootstrap_ci95": result["unit_bootstrap_ci95"],
            "unit_signflip_p_two_sided": result["unit_signflip_p_two_sided"],
        })
    bh(rows)
    effects = np.asarray([row["normalized_unit_rmse_reduction"] for row in rows])
    draws = np.asarray([effects[RNG.integers(0, len(effects), len(effects))].mean() for _ in range(50_000)])
    payload = {
        "experiment": "ppx_a5_fully_matched_six_v1",
        "comparison": "no-prior NN vs prior+same nonlinear residual",
        "rows": rows,
        "summary": {
            "normalized_mean": float(effects.mean()),
            "setting_bootstrap_ci95": np.quantile(draws, (0.025, 0.975)).tolist(),
            "positive_settings": int(np.sum(effects > 0)),
            "negative_settings": int(np.sum(effects < 0)),
            "significant_prior_benefit": int(sum(row["unit_bootstrap_ci95"][0] > 0 and row["unit_signflip_q_bh"] < 0.05 for row in rows)),
            "significant_prior_harm": int(sum(row["unit_bootstrap_ci95"][1] < 0 and row["unit_signflip_q_bh"] < 0.05 for row in rows)),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# PP-X A5 — Fully Matched Six-Setting Prior Contribution", "",
             "No-prior NN과 prior+residual에 동일 nonlinear capacity, split, seed, optimizer, budget, checkpoint rule을 적용했다.", "",
             "| Setting | No-prior R² | Prior+Residual R² | ΔR² | Unit RMSE reduction CI | q_BH |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lo, hi = row["unit_bootstrap_ci95"]
        lines.append(f"| {row['setting']} | {row['direct_r2']:.3f} | {row['prior_residual_r2']:.3f} | {row['delta_r2']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['unit_signflip_q_bh']:.3g} |")
    s = payload["summary"]
    lines += ["", "## Summary", "",
              f"- positive/negative settings: **{s['positive_settings']}/{s['negative_settings']}**",
              f"- normalized mean effect: **{s['normalized_mean']:+.3f}**",
              f"- setting-bootstrap 95% CI: **[{s['setting_bootstrap_ci95'][0]:+.3f}, {s['setting_bootstrap_ci95'][1]:+.3f}]**",
              f"- individually significant prior benefit/harm after BH: **{s['significant_prior_benefit']}/{s['significant_prior_harm']}**", "",
              "**Conclusion:** prior conditioning is heterogeneous and is not a universal performance improvement. Residual correction remains necessary, while the structural prior acts as a setting-dependent extrapolation reference.", "",
              "This is retrospective matched evidence, not prospective confirmation.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
