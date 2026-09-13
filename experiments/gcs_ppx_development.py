#!/usr/bin/env python3
"""Nested grouped cross-fitted development of safe consensus PP-X."""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from cdcr_ppx_external_cohort_replay import fit_development_head
from final_modular_pp_evidence import build_datasets, r2
from pp_extrapolation.cross_domain_consensus import (
    consensus_representation,
    fit_cross_domain_consensus_head,
    normalized_residual_target,
    predict_cross_domain_consensus,
)

OUT = ROOT / "results" / "gcs_ppx_development_v1"
PROTOCOL = "protocols/GCS_PPX_DEVELOPMENT_PROTOCOL.md"


@dataclass(frozen=True)
class Setting:
    name: str
    cohort: str
    scope: str
    y: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class JackknifeSignal:
    base: np.ndarray
    predictions: np.ndarray
    scale: float
    median: np.ndarray
    sign_agreement: np.ndarray
    iqr: np.ndarray


def load_settings() -> list[Setting]:
    settings = []
    for name, values in build_datasets().items():
        y, _groups, predictions, *_ = values
        settings.append(Setting(
            name=str(name), cohort=str(name), scope="main",
            y=np.asarray(y, dtype=np.float64),
            predictions=np.asarray(predictions, dtype=np.float64),
        ))
    external = np.load(
        ROOT / "results" / "external_cohort_four_model_comparison_v1"
        / "predictions.npz",
        allow_pickle=True,
    )
    specs = (
        ("Axial-fan 1P_8F", "axial_fan", "axial_fan_1p_8f"),
        ("Axial-fan 4P_1F", "axial_fan", "axial_fan_4p_1f"),
        ("Axial-fan 4P_8F", "axial_fan", "axial_fan_4p_8f"),
        ("MATWI tool-life", "matwi", "matwi_tool_life"),
        ("Misata machine", "misata", "misata_machine_degradation"),
    )
    for name, cohort, key in specs:
        settings.append(Setting(
            name=name, cohort=cohort, scope="external",
            y=np.asarray(external[f"{key}_truth"], dtype=np.float64),
            predictions=np.asarray(external[f"{key}_ppx"], dtype=np.float64),
        ))
    return settings


def fit_head(settings: list[Setting], alpha: float, excluded: str | None = None):
    features, targets, domains = [], [], []
    for setting in settings:
        if setting.cohort == excluded:
            continue
        representation = consensus_representation(setting.predictions)
        features.append(representation.features)
        targets.append(normalized_residual_target(setting.y, representation))
        domains.extend([setting.cohort] * len(setting.y))
    return fit_cross_domain_consensus_head(
        np.concatenate(features), np.concatenate(targets), np.asarray(domains),
        ridge_alpha=alpha,
    )


def jackknife_signal(
    train_settings: list[Setting], target: Setting, alpha: float
) -> JackknifeSignal:
    representation = consensus_representation(target.predictions)
    cohorts = sorted({setting.cohort for setting in train_settings})
    heads = [fit_head(train_settings, alpha)]
    heads.extend(fit_head(train_settings, alpha, cohort) for cohort in cohorts)
    normalized = np.asarray([
        representation.features @ head.coefficient + head.intercept
        for head in heads
    ])
    median = np.median(normalized, axis=0)
    positive = np.mean(normalized >= 0.0, axis=0)
    sign_agreement = np.maximum(positive, 1.0 - positive)
    q25, q75 = np.quantile(normalized, (0.25, 0.75), axis=0)
    return JackknifeSignal(
        base=representation.ensemble,
        predictions=target.predictions,
        scale=representation.target_scale,
        median=median,
        sign_agreement=sign_agreement,
        iqr=q75 - q25,
    )


def apply_policy(signal: JackknifeSignal, policy: dict) -> tuple[np.ndarray, np.ndarray, float]:
    stable = (
        (signal.sign_agreement >= policy["agreement"])
        & (signal.iqr <= policy["max_iqr"])
    )
    normalized = np.clip(
        signal.median, -policy["max_correction"], policy["max_correction"]
    )
    correction = policy["authority"] * signal.scale * normalized * stable
    ensemble = signal.base + correction
    seeds = ensemble + policy["shrinkage"] * (
        signal.predictions - signal.base
    )
    return ensemble, seeds, float(np.mean(stable))


def setting_scores(setting: Setting, ensemble: np.ndarray, seeds: np.ndarray) -> dict:
    values = np.asarray([r2(setting.y, prediction) for prediction in seeds])
    return {
        "r2": r2(setting.y, ensemble),
        "seed_r2_mean": float(np.mean(values)),
        "seed_r2_min": float(np.min(values)),
        "seed_r2_sd": float(np.std(values, ddof=1)),
        "seed_r2": values.tolist(),
    }


def objective(rows: list[dict], authority: float) -> tuple:
    values = np.asarray([row["r2"] for row in rows])
    return (
        int(np.sum(values > 0.0)),
        float(np.min(values)),
        float(np.quantile(values, 0.10)),
        float(np.mean(values)),
        float(np.mean([row["seed_r2_mean"] for row in rows])),
        -float(np.mean([row["seed_r2_sd"] for row in rows])),
        -float(authority),
    )


def policies() -> list[dict]:
    result = []
    for shrinkage in (0.25, 0.50):
        result.append({
            "alpha": 100.0, "authority": 0.0, "agreement": 1.0,
            "max_iqr": 0.0, "max_correction": 0.0,
            "shrinkage": shrinkage,
        })
    for alpha, authority, agreement, max_iqr, max_correction, shrinkage in itertools.product(
        (10.0, 100.0, 1000.0),
        (0.15, 0.30, 0.50),
        (0.60, 0.75, 0.90),
        (0.05, 0.15, 0.30),
        (0.10, 0.25),
        (0.25, 0.50),
    ):
        result.append({
            "alpha": alpha, "authority": authority,
            "agreement": agreement, "max_iqr": max_iqr,
            "max_correction": max_correction, "shrinkage": shrinkage,
        })
    return result


def select_policy(train_settings: list[Setting], candidates: list[dict]) -> tuple[dict, tuple]:
    cohorts = sorted({setting.cohort for setting in train_settings})
    cache = {}
    for alpha in sorted({policy["alpha"] for policy in candidates}):
        for held_out in cohorts:
            inner_train = [setting for setting in train_settings if setting.cohort != held_out]
            for setting in train_settings:
                if setting.cohort == held_out:
                    cache[(alpha, held_out, setting.name)] = jackknife_signal(
                        inner_train, setting, alpha
                    )
    best_policy, best_objective = None, None
    for policy in candidates:
        rows = []
        for held_out in cohorts:
            for setting in train_settings:
                if setting.cohort != held_out:
                    continue
                ensemble, seeds, _ = apply_policy(
                    cache[(policy["alpha"], held_out, setting.name)], policy
                )
                rows.append(setting_scores(setting, ensemble, seeds))
        value = objective(rows, policy["authority"])
        if best_objective is None or value > best_objective:
            best_policy, best_objective = policy, value
    return dict(best_policy), best_objective


def summarize(rows: list[dict], scope: str | None = None) -> dict:
    chosen = [row for row in rows if scope is None or row["scope"] == scope]
    values = np.asarray([row["gcs_ppx"]["r2"] for row in chosen])
    base = np.asarray([row["ppx"]["r2"] for row in chosen])
    return {
        "settings": len(chosen),
        "mean_setting_r2_ppx": float(np.mean(base)),
        "mean_setting_r2_gcs_ppx": float(np.mean(values)),
        "minimum_setting_r2_ppx": float(np.min(base)),
        "minimum_setting_r2_gcs_ppx": float(np.min(values)),
        "positive_settings_ppx": int(np.sum(base > 0.0)),
        "positive_settings_gcs_ppx": int(np.sum(values > 0.0)),
        "improve_tie_harm": {
            "improve": int(np.sum(values > base + 1e-12)),
            "tie": int(np.sum(np.abs(values - base) <= 1e-12)),
            "harm": int(np.sum(values < base - 1e-12)),
        },
        "mean_seed_r2_ppx": float(np.mean([
            row["ppx"]["seed_r2_mean"] for row in chosen
        ])),
        "mean_seed_r2_gcs_ppx": float(np.mean([
            row["gcs_ppx"]["seed_r2_mean"] for row in chosen
        ])),
        "minimum_seed_r2_ppx": float(np.min([
            row["ppx"]["seed_r2_min"] for row in chosen
        ])),
        "minimum_seed_r2_gcs_ppx": float(np.min([
            row["gcs_ppx"]["seed_r2_min"] for row in chosen
        ])),
        "mean_seed_sd_ppx": float(np.mean([
            row["ppx"]["seed_r2_sd"] for row in chosen
        ])),
        "mean_seed_sd_gcs_ppx": float(np.mean([
            row["gcs_ppx"]["seed_r2_sd"] for row in chosen
        ])),
    }


def markdown(payload: dict) -> str:
    lines = [
        "# GCS-PPX nested grouped cross-fit 개발 결과\n\n",
        "> 9개 기존 설정과 이미 열린 외부 5설정을 모두 개발 pool로 사용하되, "
        "각 점수는 해당 코호트 전체를 배제한 outer prediction이다.\n\n",
        "| 범위 / 설정 | PP-X R2 | GCS-PPX R2 | delta | active correction |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in payload["settings"]:
        delta = row["gcs_ppx"]["r2"] - row["ppx"]["r2"]
        lines.append(
            f"| {row['scope']} / {row['name']} | {row['ppx']['r2']:.4f} | "
            f"{row['gcs_ppx']['r2']:.4f} | {delta:+.4f} | "
            f"{row['active_fraction']:.1%} |\n"
        )
    lines.append("\n## 요약\n\n")
    for scope in ("main", "external", "all"):
        summary = payload["summary"][scope]
        lines.extend([
            f"### {scope}\n\n",
            f"- mean setting R2: PP-X {summary['mean_setting_r2_ppx']:.4f} -> "
            f"GCS-PPX {summary['mean_setting_r2_gcs_ppx']:.4f}\n",
            f"- minimum setting R2: PP-X {summary['minimum_setting_r2_ppx']:.4f} -> "
            f"GCS-PPX {summary['minimum_setting_r2_gcs_ppx']:.4f}\n",
            f"- positive settings: {summary['positive_settings_ppx']}/{summary['settings']} -> "
            f"{summary['positive_settings_gcs_ppx']}/{summary['settings']}\n",
            f"- improve/tie/harm: {summary['improve_tie_harm']}\n",
            f"- mean seed SD: {summary['mean_seed_sd_ppx']:.4f} -> "
            f"{summary['mean_seed_sd_gcs_ppx']:.4f}\n\n",
        ])
    lines.extend([
        "## 판정\n\n",
        payload["verdict"] + "\n\n",
        "이 결과는 nested grouped cross-fit 개발 증거이며, 새 untouched 외부 "
        "코호트 확증 전에는 일반적 우월성을 주장하지 않는다.\n",
    ])
    return "".join(lines)


def main() -> None:
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    settings = load_settings()
    candidates = policies()
    cohorts = sorted({setting.cohort for setting in settings})
    rows = []
    arrays = {}
    for held_out in cohorts:
        train_settings = [setting for setting in settings if setting.cohort != held_out]
        policy, inner_objective = select_policy(train_settings, candidates)
        for setting in settings:
            if setting.cohort != held_out:
                continue
            signal = jackknife_signal(train_settings, setting, policy["alpha"])
            ensemble, seeds, active = apply_policy(signal, policy)
            ppx_scores = setting_scores(
                setting, np.mean(setting.predictions, axis=0), setting.predictions
            )
            gcs_scores = setting_scores(setting, ensemble, seeds)
            rows.append({
                "name": setting.name,
                "cohort": setting.cohort,
                "scope": setting.scope,
                "outer_selected_policy": policy,
                "inner_objective": list(inner_objective),
                "active_fraction": active,
                "ppx": ppx_scores,
                "gcs_ppx": gcs_scores,
            })
            key = setting.name.lower().replace(" ", "_").replace("-", "_")
            arrays[f"{key}_truth"] = setting.y
            arrays[f"{key}_ppx"] = setting.predictions
            arrays[f"{key}_gcs_ppx"] = seeds
            print(
                "OUTER", held_out, setting.name,
                round(ppx_scores["r2"], 6), round(gcs_scores["r2"], 6),
                policy, "active", round(active, 4), flush=True,
            )
    summary = {
        "main": summarize(rows, "main"),
        "external": summarize(rows, "external"),
        "all": summarize(rows),
    }
    all_summary = summary["all"]
    success = (
        all_summary["positive_settings_gcs_ppx"]
        >= all_summary["positive_settings_ppx"]
        and all_summary["minimum_setting_r2_gcs_ppx"]
        >= all_summary["minimum_setting_r2_ppx"] - 1e-12
        and all_summary["mean_setting_r2_gcs_ppx"]
        > all_summary["mean_setting_r2_ppx"]
        and all_summary["mean_seed_sd_gcs_ppx"]
        < all_summary["mean_seed_sd_ppx"]
    )
    verdict = (
        "개발 승격: 전체 coverage와 worst-setting을 유지하면서 mean R2와 seed 안정성을 개선했다."
        if success else
        "개발 보류: coverage, worst-setting, mean R2, seed 안정성의 동시 개선 기준을 충족하지 못했다."
    )
    global_policy, global_objective = select_policy(settings, candidates)
    payload = {
        "model": "Grouped Cross-Fitted Safe Consensus PP-X (GCS-PPX)",
        "status": "retrospective nested grouped cross-fit development",
        "protocol": PROTOCOL,
        "candidate_policies": len(candidates),
        "cohort_groups": cohorts,
        "settings": rows,
        "summary": summary,
        "development_success": success,
        "verdict": verdict,
        "deployment_policy_selected_from_all_opened_cohorts": global_policy,
        "deployment_inner_objective": list(global_objective),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "cross_fitted_predictions.npz", **arrays)
    (OUT / "RESULTS_KO.md").write_text(markdown(payload))
    print(json.dumps({"summary": summary, "success": success,
                      "global_policy": global_policy}, indent=2), flush=True)


if __name__ == "__main__":
    main()
