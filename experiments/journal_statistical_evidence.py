#!/usr/bin/env python3
"""Multiplicity-aware and hierarchical evidence summary for matched PP vs MLP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "cross_domain_mechanism_v2"
GEOMETRY = ROOT / "results" / "journal_support_geometry_v1" / "results.json"
ROUTE = ROOT / "results" / "journal_validation_route_v1" / "results.json"
OUTPUT = ROOT / "results" / "journal_statistical_evidence_v1"
RNG = np.random.default_rng(20260908)


def exact_signflip(delta: np.ndarray) -> float:
    n = len(delta)
    observed = abs(float(delta.mean()))
    if n <= 20:
        masks = np.arange(1 << n, dtype=np.uint64)[:, None]
        signs = 2 * (((masks >> np.arange(n, dtype=np.uint64)) & 1).astype(float)) - 1
        return float(np.mean(np.abs(signs @ delta / n) >= observed - 1e-15))
    signs = RNG.choice((-1.0, 1.0), size=(200000, n))
    return float((1 + np.sum(np.abs(signs @ delta / n) >= observed)) / (len(signs) + 1))


def bh_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def unit_delta(path: Path) -> tuple[np.ndarray, list[str]]:
    saved = np.load(path, allow_pickle=True)
    y = np.asarray(saved["y"], float)
    groups = np.asarray(saved["groups"])
    plain = saved["plain"].mean(0)
    pp = saved["pp"].mean(0)
    delta, units = [], []
    for unit in np.unique(groups):
        mask = groups == unit
        a = np.sqrt(np.mean((y[mask] - plain[mask]) ** 2))
        b = np.sqrt(np.mean((y[mask] - pp[mask]) ** 2))
        delta.append((a - b) / max(a, 1e-10))
        units.append(str(unit))
    return np.asarray(delta), units


def hierarchical_bootstrap(deltas: list[np.ndarray], n: int = 50000) -> dict:
    estimates = np.empty(n)
    medians = np.empty(n)
    for b in range(n):
        selected = RNG.integers(0, len(deltas), len(deltas))
        domain_effects = []
        for index in selected:
            value = deltas[index]
            domain_effects.append(float(np.mean(RNG.choice(value, len(value), replace=True))))
        estimates[b] = np.mean(domain_effects)
        medians[b] = np.median(domain_effects)
    domain_means = np.asarray([value.mean() for value in deltas])
    return {
        "equal_dataset_mean": float(domain_means.mean()),
        "equal_dataset_mean_ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "median_dataset_effect": float(np.median(domain_means)),
        "median_dataset_effect_ci95": np.quantile(medians, [0.025, 0.975]).tolist(),
        "probability_bootstrap_mean_positive": float(np.mean(estimates > 0)),
    }


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    geometry = json.loads(GEOMETRY.read_text())
    route = json.loads(ROUTE.read_text())
    rows, deltas = [], []
    for name in metadata["datasets"]:
        delta, units = unit_delta(SOURCE / f"{name}_predictions.npz")
        row = {
            "dataset": name,
            "n_units": len(delta),
            "unit_wins": int(np.sum(delta > 0)),
            "mean_relative_rmse_gain": float(delta.mean()),
            "median_relative_rmse_gain": float(np.median(delta)),
            "unit_bootstrap_ci95": np.quantile(
                RNG.choice(delta, size=(50000, len(delta)), replace=True).mean(1), [0.025, 0.975]
            ).tolist(),
            "paired_signflip_p": exact_signflip(delta),
            "unit_ids": units,
            "unit_relative_rmse_gain": delta.tolist(),
        }
        rows.append(row)
        deltas.append(delta)
    adjusted = bh_adjust([row["paired_signflip_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["bh_fdr_adjusted_p"] = value

    means = np.asarray([row["mean_relative_rmse_gain"] for row in rows])
    positive = int(np.sum(means > 0))
    strict_names = []
    for row in geometry["dataset_rows"]:
        if row["dataset"] == "nasa_battery":
            strict_names.append(row["dataset"])
        elif row["dataset"] in geometry["datasets"] and "support" in geometry["datasets"][row["dataset"]]:
            fraction = geometry["datasets"][row["dataset"]]["support"]["ordered_1d_hull"]["outside_fraction"]
            if fraction >= 0.95:
                strict_names.append(row["dataset"])
    strict_rows = [row for row in rows if row["dataset"] in strict_names]
    strict_positive = sum(row["mean_relative_rmse_gain"] > 0 for row in strict_rows)

    # Threshold sweep is a diagnostic only. It demonstrates whether increasing
    # the required validation gain removes false accepts; no threshold is chosen.
    threshold_sweep = []
    for threshold in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        accepted = [row for row in route["datasets"] if row["validation_relative_gain"] >= threshold]
        false_accepts = sum(not row["test_pp_helped"] for row in accepted)
        route_effect = [row["test_pp_mean_relative_unit_gain"] if row in accepted else 0.0 for row in route["datasets"]]
        threshold_sweep.append({
            "minimum_validation_relative_gain": threshold,
            "accepted": len(accepted),
            "false_accepts": false_accepts,
            "false_rejects": sum(row["test_pp_helped"] and row not in accepted for row in route["datasets"]),
            "equal_dataset_route_gain": float(np.mean(route_effect)),
            "accepted_datasets": [row["dataset"] for row in accepted],
        })

    result = {
        "experiment": "journal_statistical_evidence_v1",
        "status": "retrospective matched common-backbone audit",
        "effect": "positive values mean PP reduces physical-unit RMSE relative to matched MLP",
        "all_datasets": {
            "n": len(rows),
            "dataset_wins": positive,
            "two_sided_dataset_sign_test_p": float(binomtest(positive, len(rows), 0.5).pvalue),
            "hierarchical_bootstrap": hierarchical_bootstrap(deltas),
        },
        "strict_ordered_coordinate_subgroup": {
            "definition": ">=95% of test rows outside train range on declared first degradation coordinate",
            "datasets": strict_names,
            "n": len(strict_rows),
            "dataset_wins": strict_positive,
            "two_sided_dataset_sign_test_p": float(binomtest(strict_positive, len(strict_rows), 0.5).pvalue),
            "hierarchical_bootstrap": hierarchical_bootstrap(
                [deltas[i] for i, row in enumerate(rows) if row["dataset"] in strict_names]
            ),
        },
        "multiplicity_aware_dataset_tests": rows,
        "validation_gain_threshold_sensitivity": threshold_sweep,
        "conclusion_guardrails": [
            "The common PP backbone is not universally better than the matched MLP across all domains.",
            "A validation-gain threshold alone does not eliminate false PP accepts; milling remains a mechanism-shift counterexample even at 50% validation gain.",
            "Dataset-specific final PP variants are development-best results and must be separated from this common-backbone confirmatory-style comparison.",
            "BH adjustment treats datasets as a reported family; it does not make heterogeneous domains exchangeable.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "all": result["all_datasets"],
        "strict": result["strict_ordered_coordinate_subgroup"],
        "threshold_sweep": threshold_sweep,
    }, indent=2))


if __name__ == "__main__":
    main()
