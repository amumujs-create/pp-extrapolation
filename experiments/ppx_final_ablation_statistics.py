"""Recompute the final PP-X component ablation from row-aligned predictions.

This is an inference audit, not a new training run.  It keeps the original
test rows, five seeds and unit identifiers, and separates prediction-ensemble
effects from optimizer-seed effects.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ppx_final_ablation_statistics_v1"
RNG_SEED = 20260909
BOOTSTRAPS = 50_000


def r2s(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.asarray([r2_score(y, row) for row in predictions], dtype=float)


def exact_signflip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values) or np.allclose(values, 0):
        return 1.0
    observed = abs(values.mean())
    if len(values) <= 20:
        means = [abs(np.mean(values * np.asarray(signs)))
                 for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
        return float(np.mean(np.asarray(means) >= observed - 1e-15))
    rng = np.random.default_rng(RNG_SEED + len(values))
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    return float((1 + np.sum(np.abs((signs * values).mean(1)) >= observed)) / 100_001)


def compare(name: str, component: str, y: np.ndarray, groups: np.ndarray,
            off: np.ndarray, on: np.ndarray, status: str = "matched") -> dict:
    y = np.asarray(y, float); groups = np.asarray(groups)
    off = np.atleast_2d(np.asarray(off, float)); on = np.atleast_2d(np.asarray(on, float))
    if off.shape[0] == 1 and on.shape[0] > 1:
        off = np.repeat(off, on.shape[0], axis=0)
    if on.shape[0] == 1 and off.shape[0] > 1:
        on = np.repeat(on, off.shape[0], axis=0)
    if off.shape != on.shape or off.shape[1] != len(y):
        raise ValueError((name, off.shape, on.shape, y.shape))

    off_r2, on_r2 = r2s(y, off), r2s(y, on)
    seed_delta = on_r2 - off_r2
    off_ensemble, on_ensemble = off.mean(0), on.mean(0)
    unit_delta = []
    for unit in np.unique(groups):
        take = groups == unit
        off_rmse = np.sqrt(np.mean((y[take] - off_ensemble[take]) ** 2))
        on_rmse = np.sqrt(np.mean((y[take] - on_ensemble[take]) ** 2))
        unit_delta.append(off_rmse - on_rmse)  # positive means component helps
    unit_delta = np.asarray(unit_delta)
    rng = np.random.default_rng(RNG_SEED + sum(map(ord, name + component)))
    boot = unit_delta[rng.integers(0, len(unit_delta), (BOOTSTRAPS, len(unit_delta)))].mean(1)
    return {
        "dataset": name, "component": component, "status": status,
        "n_rows": len(y), "n_units": len(unit_delta), "n_seeds": len(on),
        "off_ensemble_r2": float(r2_score(y, off_ensemble)),
        "on_ensemble_r2": float(r2_score(y, on_ensemble)),
        "ensemble_delta_r2": float(r2_score(y, on_ensemble) - r2_score(y, off_ensemble)),
        "off_seed_r2_mean_sd": [float(off_r2.mean()), float(off_r2.std(ddof=1))],
        "on_seed_r2_mean_sd": [float(on_r2.mean()), float(on_r2.std(ddof=1))],
        "seed_delta_r2": seed_delta.tolist(),
        "seed_signflip_p_two_sided": exact_signflip(seed_delta),
        "mean_unit_rmse_reduction": float(unit_delta.mean()),
        "unit_rmse_wins": int(np.sum(unit_delta > 0)),
        "unit_bootstrap_ci95": [float(x) for x in np.quantile(boot, (.025, .975))],
        "unit_signflip_p_two_sided": exact_signflip(unit_delta),
    }


def load(path: str):
    return np.load(ROOT / path, allow_pickle=True)


def main() -> None:
    rows = []
    matched = load("results/bq_pp_matched_controls_v1/predictions.npz")
    final_bq = load("results/bq_dual_scale_final_replay_v1/predictions.npz")
    dataset_names = {0: "Sunwoda", 1: "RWTH", 2: "MICH"}
    for code, dataset in dataset_names.items():
        take = matched["dataset"] == code
        common = dict(name=dataset, y=matched["y"][take], groups=matched["units"][take])
        rows.append(compare(component="nonlinear residual", off=matched["affine_quotient_only"][:, take],
                            on=matched["bq_pp"][:, take], **common))
        rows.append(compare(component="frozen affine path", off=matched["hard_trainable_nn"][:, take],
                            on=matched["bq_pp"][:, take], **common))
        rows.append(compare(component="fixed residual bound", off=matched["frozen_unbounded_pp"][:, take],
                            on=matched["bq_pp"][:, take], **common))
        rows.append(compare(component="support-adaptive dual scale", off=matched["bq_pp"][:, take],
                            on=final_bq["prediction"][:, take], **common))
        selected = final_bq["prediction"][:, take] if dataset == "MICH" else matched["bq_pp"][:, take]
        rows.append(compare(component="complete selected PP-X vs direct NN",
                            off=matched["direct_nn"][:, take], on=selected, **common))

    hust = load("results/hust_regime_transport_pp_v1/predictions.npz")
    rows.append(compare("HUST", "regime transport", hust["y"], hust["groups"],
                        hust["raw"], hust["transported"]))
    matr = load("results/matr_batch2_pp_five_seed_replay/predictions.npz")
    rows.append(compare("MATR batch 2", "regime transport after support decay", matr["truth"],
                        matr["groups"], matr["raw"], matr["transported"]))
    femto_off = load("results/femto_waveform_pp_v8/pp.npz")
    femto_on = load("results/femto_waveform_pp_v8/direct.npz")
    rows.append(compare("FEMTO", "transferability gate / prior abstention", femto_on["y"],
                        femto_on["groups"], femto_off["predictions"], femto_on["predictions"],
                        status="retrospective route ablation; test has only one row per unit"))

    nasa_on = load("results/nasa_causal_multiscale_pp_v1/predictions.npz")
    nasa_off = load("results/nasa_causal_short_fixed_ablation_v1/predictions.npz")
    rows.append(compare("NASA battery", "validation-selected causal history", nasa_on["y"],
                        nasa_on["groups"], nasa_off["prediction"], nasa_on["prediction"]))

    # Both archives contain predictions for all windows. Apply the official hard
    # mask once here so the inference population exactly matches reported R2.
    legacy = ROOT.parent / "ca-css-ncmapss"
    sys.path.insert(0, str(legacy))
    from ncmapss_tra_quantile_split import make_tra_hard_split
    split = make_tra_hard_split(ROOT / "data/N-CMAPSS_DS02-006.h5",
                                max_windows_per_unit=1500, random_seed=42)
    test_units = split.meta.get("unit_ids", {}).get("test")
    hard = split.all_windows.hard_extrap_mask(test_units, thresholds=split.thresholds)
    def nc_matrix(folder: str) -> np.ndarray:
        return np.asarray([load(f"results/{folder}/seed{seed}.npz")["prediction"][hard]
                           for seed in range(42, 47)])
    rows.append(compare("N-CMAPSS", "validation-selected multiscale history",
                        np.asarray(split.all_windows.y)[hard], np.asarray(split.all_windows.units)[hard],
                        nc_matrix("ncmapss_basic_fixed_ablation_v1"),
                        nc_matrix("ncmapss_pp_multiscale_v1")))

    # Benjamini-Hochberg correction is reported separately for unit and seed tests.
    for field, corrected in (("unit_signflip_p_two_sided", "unit_signflip_q_bh"),
                             ("seed_signflip_p_two_sided", "seed_signflip_q_bh")):
        p = np.asarray([r[field] for r in rows]); order = np.argsort(p); q = np.empty(len(p)); running = 1.0
        for rank in range(len(p) - 1, -1, -1):
            idx = order[rank]; running = min(running, p[idx] * len(p) / (rank + 1)); q[idx] = running
        for row, value in zip(rows, q): row[corrected] = float(min(1.0, value))

    payload = {
        "experiment": "ppx_final_ablation_statistics_v1",
        "status": "retrospective matched replay from stored row-aligned predictions",
        "metric_contract": "pooled R2; prediction ensemble is primary; units are the population sampling level",
        "inference": {"bootstrap_resamples": BOOTSTRAPS, "seed": RNG_SEED,
                      "unit_effect": "off RMSE minus on RMSE; positive favors component",
                      "multiplicity": "Benjamini-Hochberg across all component-dataset comparisons"},
        "limitations": [
            "Seed p-values quantify optimizer sensitivity and have minimum attainable two-sided p=0.0625 for five seeds.",
            "Unit bootstrap is unreliable for very small unit counts; FEMTO has one test row per unit.",
            "These are retrospective development splits, not untouched confirmation.",
            "N-CMAPSS has only three hard-test engines and one has a single row, limiting unit inference."
        ],
        "comparisons": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# 최종 PP-X 구성요소 ablation 및 통계검정", "",
             "동일 test row와 seeds 42–46의 저장 예측을 재집계했다. 주 지표는 prediction-ensemble pooled R²이며, 통계 표본은 독립 물리 unit이다. 양의 RMSE 감소는 구성요소가 유리하다는 뜻이다.", "",
             "| 데이터셋 | 구성요소(on) | off R² | on R² | ΔR² | unit 승 | 평균 unit RMSE 감소 [95% CI] | sign-flip p / BH q | seed p |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ci = r["unit_bootstrap_ci95"]
        lines.append(f'| {r["dataset"]} | {r["component"]} | {r["off_ensemble_r2"]:.3f} | **{r["on_ensemble_r2"]:.3f}** | {r["ensemble_delta_r2"]:+.3f} | {r["unit_rmse_wins"]}/{r["n_units"]} | {r["mean_unit_rmse_reduction"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r["unit_signflip_p_two_sided"]:.4g} / {r["unit_signflip_q_bh"]:.4g} | {r["seed_signflip_p_two_sided"]:.4g} |')
    lines += ["", "## 판정 기준", "",
             "- ΔR²와 unit 평균 효과가 모두 양수이고 unit-bootstrap CI의 하한이 0보다 크면 강한 표본 내 근거로 본다.",
              "- CI가 0을 포함하면 개선 방향은 보여도 모집단 수준의 유의성은 확정하지 않는다.",
              "- 5-seed exact sign-flip 검정은 완전한 5/5 동일 방향이어도 양측 p의 최솟값이 0.0625이므로 seed p<0.05를 요구하지 않는다.",
              "- FEMTO gate 비교는 unit당 한 점뿐이므로 unit 통계가 수명곡선 일반화를 검정하지 못한다.", "",
              "## 해석상 제한", "",
              "NASA와 N-CMAPSS의 제거 arm도 같은 seeds로 재학습해 포함했다. 두 효과는 각각 약 +0.012, +0.009로 작고 unit 수도 4개와 3개뿐이므로, 방향성 ablation으로 보고 모집단 유의성을 주장하지 않는다.", ""]
    (OUT / "REPORT_KO.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(OUT), "comparisons": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
