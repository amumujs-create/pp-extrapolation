#!/usr/bin/env python3
"""Post-hoc validation audit for CCMR v2.2 development domains only.

This script does not refit or select a new threshold. It quantifies:
1) physical-unit paired v2.2-vs-v2.0 effects,
2) false accept / false reject proxies, and
3) frozen route-threshold sensitivity as a diagnostic grid.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V20 = ROOT / "results/ccmr_v20_trajectory_development/results.json"
V22 = ROOT / "results/ccmr_v22_trajectory_development/results.json"
OUT = ROOT / "results/ccmr_v22_validation_gap_audit"
RNG = np.random.default_rng(20260911)


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not np.any(np.abs(values) > 1e-15):
        return 1.0
    observed = abs(float(np.mean(values)))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.abs(signs @ values / len(values))
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_ci(values: np.ndarray, reps: int = 50_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    draws = RNG.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, (0.025, 0.975)).tolist()


def hierarchical_bootstrap(domain_values: list[np.ndarray], reps: int = 50_000):
    estimates = np.empty(reps)
    for i in range(reps):
        domain_index = RNG.integers(0, len(domain_values), len(domain_values))
        means = []
        for index in domain_index:
            values = domain_values[index]
            means.append(float(np.mean(RNG.choice(values, len(values), replace=True))))
        estimates[i] = np.mean(means)
    return {
        "mean_equal_domain_relative_rmse_gain": float(
            np.mean([np.mean(values) for values in domain_values])
        ),
        "ci95": np.quantile(estimates, (0.025, 0.975)).tolist(),
    }


def unit_paired(v20: dict, v22: dict):
    rows = []
    domain_values = []
    for name in v22["cohorts"]:
        old = v20["cohorts"][name]["test"]["model"]["per_unit"]
        new = v22["cohorts"][name]["test"]["model"]["per_unit"]
        units = sorted(set(old) & set(new))
        if set(old) != set(new):
            raise RuntimeError(f"{name}: v2.0/v2.2 unit IDs do not align")
        old_rmse = np.asarray([old[unit]["rmse"] for unit in units])
        new_rmse = np.asarray([new[unit]["rmse"] for unit in units])
        relative = (old_rmse - new_rmse) / np.maximum(old_rmse, 1e-12)
        domain_values.append(relative)
        rows.append({
            "domain": name,
            "n_units": len(units),
            "v22_unit_wins": int(np.sum(relative > 1e-12)),
            "ties": int(np.sum(np.abs(relative) <= 1e-12)),
            "mean_relative_rmse_gain": float(np.mean(relative)),
            "median_relative_rmse_gain": float(np.median(relative)),
            "unit_bootstrap_ci95": bootstrap_ci(relative),
            "paired_sign_flip_p": exact_sign_flip(relative),
            "unit_ids": units,
            "unit_relative_rmse_gain": relative.tolist(),
        })
    domain_means = np.asarray([row["mean_relative_rmse_gain"] for row in rows])
    return {
        "datasets": rows,
        "aggregate": {
            "domains": len(rows),
            "physical_units": int(sum(row["n_units"] for row in rows)),
            "domains_positive": int(np.sum(domain_means > 1e-12)),
            "domain_sign_flip_p": exact_sign_flip(domain_means),
            "domain_bootstrap_ci95": bootstrap_ci(domain_means),
            "hierarchical_bootstrap": hierarchical_bootstrap(domain_values),
        },
    }


def confusion_audit(v22: dict):
    rows = []
    for name, value in v22["cohorts"].items():
        approved = bool(value["route"]["approved"])
        deployed_gain = float(value["test"]["pooled_improvement"])
        raw_gain = float(value["ablation"]["v22_raw_bank"]["pooled_improvement"])
        rows.append({
            "domain": name,
            "approved": approved,
            "route": value["route"]["route"],
            "deployed_pooled_improvement": deployed_gain,
            "raw_bank_pooled_improvement": raw_gain,
            "false_accept": bool(approved and deployed_gain < -1e-12),
            # Diagnostic proxy only: an unapproved raw bank has positive pooled
            # gain, regardless of whether its unit-risk profile is acceptable.
            "pooled_false_reject_proxy": bool(
                not approved and raw_gain > 1e-12
            ),
            "unsafe_raw_bank_rejection": bool(
                not approved
                and value["ablation"]["v22_raw_bank"]["raw_regret"]["maximum"] > 0.02
            ),
        })
    return {
        "definition": {
            "false_accept": "approved and deployed pooled improvement < 0",
            "false_reject_proxy": (
                "not approved and raw-bank pooled improvement > 0; "
                "does not imply the rejected bank was safe"
            ),
        },
        "rows": rows,
        "summary": {
            "false_accepts": sum(row["false_accept"] for row in rows),
            "pooled_false_reject_proxies": sum(
                row["pooled_false_reject_proxy"] for row in rows
            ),
            "unsafe_raw_bank_rejections": sum(
                row["unsafe_raw_bank_rejection"] for row in rows
            ),
        },
    }


def threshold_sensitivity(v22: dict):
    """Replay stable/small bank certificates without choosing a new threshold."""
    rows = []
    for active_min in (0.40, 0.50, 0.60):
        for stable_gain in (0.025, 0.05, 0.10):
            for small_gain in (0.05, 0.10, 0.15):
                accepted = []
                false_accepts = 0
                false_rejects = 0
                unsafe_accepts = 0
                route_effect = []
                for name, value in v22["cohorts"].items():
                    model = value["model"]
                    units = value["validation_units"]
                    risk_ok = (
                        model["validation_raw_regret"]["mean"] <= 0.0
                        and model["validation_raw_regret"]["cvar20"] <= 0.01
                        and model["validation_raw_regret"]["maximum"] <= 0.02
                    )
                    gain_min = stable_gain if units >= 10 else small_gain
                    certificate = bool(
                        units >= 3
                        and model["validation_active_fraction"] >= active_min
                        and model["validation_macro_improvement"] >= gain_min
                        and model["deployment_mass"] > 0
                        and risk_ok
                        and (
                            units < 10
                            or value["validation_causal_coverage"] >= 0.10
                        )
                    )
                    raw = value["ablation"]["v22_raw_bank"]
                    raw_gain = float(raw["pooled_improvement"])
                    raw_safe = raw["raw_regret"]["maximum"] <= 0.02
                    if certificate:
                        accepted.append(name)
                        false_accepts += raw_gain < -1e-12
                        unsafe_accepts += not raw_safe
                        route_effect.append(raw_gain)
                    else:
                        false_rejects += raw_gain > 1e-12 and raw_safe
                        route_effect.append(0.0)
                rows.append({
                    "active_fraction_min": active_min,
                    "stable_macro_gain_min": stable_gain,
                    "small_macro_gain_min": small_gain,
                    "accepted": len(accepted),
                    "accepted_domains": accepted,
                    "false_accepts": int(false_accepts),
                    "safe_gain_false_rejects": int(false_rejects),
                    "unsafe_accepts": int(unsafe_accepts),
                    "equal_domain_mean_pooled_improvement": float(np.mean(route_effect)),
                    "is_frozen_v22_setting": bool(
                        active_min == 0.50
                        and stable_gain == 0.05
                        and small_gain == 0.10
                    ),
                })
    return {
        "status": "diagnostic only; no threshold is selected from this grid",
        "rows": rows,
    }


def main():
    v20 = json.loads(V20.read_text())
    v22 = json.loads(V22.read_text())
    if v20["holdouts_loaded"] or v22["holdouts_loaded"]:
        raise RuntimeError("development audit must not load holdouts")
    payload = {
        "status": "CCMR v2.2 validation-gap audit complete",
        "owner": "박진서",
        "holdouts_loaded": False,
        "unit_paired_v22_vs_v20": unit_paired(v20, v22),
        "gate_confusion": confusion_audit(v22),
        "route_threshold_sensitivity": threshold_sensitivity(v22),
        "guardrails": [
            "The sweep is diagnostic and cannot be used to retune frozen v2.2.",
            "Physical units, not rows, are inference units.",
            "The five development domains are retrospective, not prospective confirmation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "unit_aggregate": payload["unit_paired_v22_vs_v20"]["aggregate"],
        "gate_confusion": payload["gate_confusion"]["summary"],
        "frozen_threshold": next(
            row for row in payload["route_threshold_sensitivity"]["rows"]
            if row["is_frozen_v22_setting"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
