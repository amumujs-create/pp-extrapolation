#!/usr/bin/env python3
"""Compare one globally fixed executor policy with the PP-X v4 router.

An executor that is not contract-computable for a setting falls back to that
setting's matched prior+residual core.  Thus each fixed policy differs from the
router only where its named executor is admissible.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ppx_final_ablation_statistics import compare  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

OUT = ROOT / "results/ppx_global_route_vs_hierarchical_v1"
REPORT = ROOT / "PPX_GLOBAL_ROUTE_VS_HIERARCHICAL_V1_KO.md"
POLICIES = ("core", "bounded", "dual_scale", "transport", "history")
FINAL_BQ_ARM = {"Sunwoda": "bounded", "RWTH": "bounded", "MICH": "dual_scale"}


@dataclass
class Setting:
    name: str
    y: np.ndarray
    groups: np.ndarray
    ref_correction: np.ndarray
    val_selected: np.ndarray
    fixed_policies: dict[str, np.ndarray]


def ens_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, np.asarray(prediction).mean(0)))


def load_bq3_settings() -> list[Setting]:
    z = np.load(ROOT / "results/ppx_main_ablation_bq3_v1/bundle.npz", allow_pickle=True)
    out = []
    for code, name in enumerate(("Sunwoda", "RWTH", "MICH")):
        take = z["test_dataset"] == code
        policies = {
            key: np.asarray(z[f"test_{key}"])[:, take]
            for key in ("unbounded", "bounded", "dual_scale")
        }
        out.append(Setting(
            name=name,
            y=z["test_y"][take],
            groups=z["test_units"][take],
            ref_correction=policies["bounded"],
            val_selected=policies[FINAL_BQ_ARM[name]],
            fixed_policies=policies,
        ))
    return out


def load_pp_npz(stem: str, name: str) -> Setting:
    z = np.load(ROOT / f"results/ppx_main_ablation_6setting_v1/{stem}.npz", allow_pickle=True)
    policies = {"transport": np.asarray(z["test_transport"])}
    if stem == "hust":
        policies["pp_core"] = np.asarray(z["test_pp_core"])
        core = policies["pp_core"]
    else:
        policies["raw_decay0"] = np.asarray(z["test_raw_decay0"])
        policies["raw_decay005"] = np.asarray(z["test_pp_core"])
        core = policies["raw_decay005"]
    return Setting(name, z["test_y"], z["test_groups"], core, policies["transport"], policies)


def load_ncmapss() -> Setting:
    z = np.load(ROOT / "results/ppx_main_ablation_6setting_v1/ncmapss.npz", allow_pickle=True)
    policies = {
        key: np.asarray(z[f"test_{key}"])
        for key in ("basic", "moments", "multiscale")
    }
    return Setting(
        "N-CMAPSS", z["test_y"], z["test_groups"], policies["multiscale"],
        np.asarray(z["test_val_selected"]), policies,
    )


def load_settings():
    values = load_bq3_settings()
    values.append(load_pp_npz("hust", "HUST"))
    values.append(load_pp_npz("matr", "MATR batch 2"))
    values.append(load_ncmapss())
    return values


def fixed_prediction(setting, policy: str) -> tuple[np.ndarray, bool]:
    """Return fixed-policy prediction and whether the executor was admissible."""
    if policy == "core":
        return setting.ref_correction, True
    if policy == "bounded" and setting.name in {"Sunwoda", "RWTH", "MICH"}:
        return setting.fixed_policies["bounded"], True
    if policy == "dual_scale" and setting.name in {"Sunwoda", "RWTH", "MICH"}:
        return setting.fixed_policies["dual_scale"], True
    if policy == "transport" and setting.name in {"HUST", "MATR batch 2"}:
        return setting.fixed_policies["transport"], True
    if policy == "history" and setting.name == "N-CMAPSS":
        return setting.val_selected, True
    return setting.ref_correction, False


def summarize_policy(settings, policy: str) -> dict:
    rows = []
    normalized = []
    for setting in settings:
        fixed, admissible = fixed_prediction(setting, policy)
        router = setting.val_selected
        stats = compare(
            setting.name,
            f"hierarchical router vs global {policy}",
            setting.y,
            setting.groups,
            fixed,
            router,
            status="global_fixed_route_control",
        )
        fixed_rmse = float(np.sqrt(np.mean((setting.y - fixed.mean(0)) ** 2)))
        norm = float(stats["mean_unit_rmse_reduction"] / max(fixed_rmse, 1e-12))
        normalized.append(norm)
        rows.append(
            {
                "setting": setting.name,
                "executor_admissible": admissible,
                "fixed_r2": ens_r2(setting.y, fixed),
                "router_r2": ens_r2(setting.y, router),
                "delta_r2_router_minus_fixed": ens_r2(setting.y, router) - ens_r2(setting.y, fixed),
                "normalized_unit_rmse_reduction": norm,
                "unit_bootstrap_ci95": stats["unit_bootstrap_ci95"],
            }
        )
    delta = np.asarray([row["delta_r2_router_minus_fixed"] for row in rows])
    admissible_rows = [row for row in rows if row["executor_admissible"]]
    admissible_delta = np.asarray(
        [row["delta_r2_router_minus_fixed"] for row in admissible_rows]
    )
    return {
        "policy": policy,
        "mean_r2": float(np.mean([row["fixed_r2"] for row in rows])),
        "worst_setting_r2": float(np.min([row["fixed_r2"] for row in rows])),
        "router_mean_r2": float(np.mean([row["router_r2"] for row in rows])),
        "router_worst_setting_r2": float(np.min([row["router_r2"] for row in rows])),
        "router_wins_ties_losses": [
            int(np.sum(delta > 1e-10)),
            int(np.sum(np.abs(delta) <= 1e-10)),
            int(np.sum(delta < -1e-10)),
        ],
        "admissible_coverage": len(admissible_rows),
        "core_fallback_count": len(rows) - len(admissible_rows),
        "admissible_subset_router_wins_ties_losses": [
            int(np.sum(admissible_delta > 1e-10)),
            int(np.sum(np.abs(admissible_delta) <= 1e-10)),
            int(np.sum(admissible_delta < -1e-10)),
        ],
        "admissible_subset_mean_delta_r2": float(np.mean(admissible_delta)),
        "mean_normalized_unit_rmse_reduction": float(np.mean(normalized)),
        "rows": rows,
    }


def write_report(payload: dict) -> None:
    lines = [
        "# Global Fixed Route vs PP-X v4 Hierarchical Router", "",
        "동일한 6개 setting과 저장된 5-seed 예측을 사용했다. 고정 executor가 contract상",
        "계산 불가능한 setting에서는 matched prior+residual core를 사용했다.", "",
        "| Fixed typed policy | Coverage | Core fallback | Overall W/T/L | Admissible W/T/L | Mean R² | Worst R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["policies"]:
        w, t, l = item["router_wins_ties_losses"]
        lines.append(
            f"| {item['policy']} | {item['admissible_coverage']}/6 | {item['core_fallback_count']} | "
            f"{w}/{t}/{l} | {'/'.join(map(str, item['admissible_subset_router_wins_ties_losses']))} | "
            f"{item['mean_r2']:.3f} | {item['worst_setting_r2']:.3f} |"
        )
    lines += ["", f"Hierarchical router mean R²: **{payload['router_mean_r2']:.3f}**", "",
              f"Hierarchical router worst-setting R²: **{payload['router_worst_setting_r2']:.3f}**", "",
              "## Setting-level ΔR² (router - fixed)", ""]
    for item in payload["policies"]:
        values = ", ".join(
            f"{row['setting']}={row['delta_r2_router_minus_fixed']:+.3f}"
            for row in item["rows"]
        )
        lines.append(f"- **{item['policy']}**: {values}")
    lines += ["", "## Interpretation", "",
              "- 한 executor를 모든 setting에 고정하면 해당 executor가 필요한 setting만 개선하고 나머지는 core에 머문다.",
              "- 계층 router는 admissible family 안에서 setting별 validation evidence로 서로 다른 executor를 선택한다.",
              "- 이 결과는 retrospective stored-prediction comparison이며 새 prospective cohort 검증은 아니다.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = load_settings()
    policies = [summarize_policy(settings, policy) for policy in POLICIES]
    payload = {
        "experiment": "ppx_global_route_vs_hierarchical_v1",
        "n_settings": len(settings),
        "router_mean_r2": policies[0]["router_mean_r2"],
        "router_worst_setting_r2": policies[0]["router_worst_setting_r2"],
        "policies": policies,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    print(json.dumps({
        "router_mean_r2": payload["router_mean_r2"],
        "router_worst_setting_r2": payload["router_worst_setting_r2"],
        "fixed_policies": [{
            "policy": item["policy"],
            "mean_r2": item["mean_r2"],
            "worst_r2": item["worst_setting_r2"],
            "router_wtl": item["router_wins_ties_losses"],
            "normalized_gain": item["mean_normalized_unit_rmse_reduction"],
        } for item in policies],
    }, indent=2))


if __name__ == "__main__":
    main()
