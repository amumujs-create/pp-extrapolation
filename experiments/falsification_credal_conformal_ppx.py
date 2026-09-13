#!/usr/bin/env python3
"""Retrospective audit of falsification-calibrated credal conformal PP-X."""
from __future__ import annotations

import json
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.coverage_preserving_credal import (  # noqa: E402
    fit_credal_conformal_ppx,
    legacy_candidate_interval,
    predict_credal_conformal_ppx,
)
from pp_extrapolation.prior_falsification import unit_log_regret  # noqa: E402

SOURCE = ROOT / "results" / "cross_domain_mechanism_v2"
OUT = ROOT / "results" / "falsification_credal_conformal_ppx_v2"
PROTOCOL = "protocols/FALSIFICATION_CREDAL_CONFORMAL_PPX_PROTOCOL.md"


def unit_effect(y, groups, fallback, prediction):
    return -unit_log_regret(y, groups, fallback, prediction)


def load_domain(name):
    saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
    validation_y = np.asarray(saved["validation_y"], dtype=np.float64)
    validation_groups = np.asarray(saved["validation_groups"]).astype(str)
    validation_fallback = np.asarray(
        saved["validation_plain"], dtype=np.float64
    ).mean(0)
    validation_candidate = np.asarray(
        saved["validation_pp"], dtype=np.float64
    ).mean(0)
    model = fit_credal_conformal_ppx(
        validation_y,
        validation_groups,
        validation_fallback,
        validation_candidate,
        saved["validation_distance"],
        seed=20260913 + zlib.crc32(name.encode()),
    )

    y = np.asarray(saved["y"], dtype=np.float64)
    groups = np.asarray(saved["groups"]).astype(str)
    fallback = np.asarray(saved["plain"], dtype=np.float64).mean(0)
    candidate = np.asarray(saved["pp"], dtype=np.float64).mean(0)
    distance = np.asarray(saved["test_distance"], dtype=np.float64)
    prediction = predict_credal_conformal_ppx(
        model, fallback, candidate, distance
    )
    legacy_lower, legacy_upper = legacy_candidate_interval(
        model, candidate, distance
    )
    legacy_covered = (y >= legacy_lower) & (y <= legacy_upper)
    updated_covered = (y >= prediction.lower) & (y <= prediction.upper)
    effect = unit_effect(y, groups, fallback, prediction.point)
    always_pp_effect = unit_effect(y, groups, fallback, candidate)
    legacy_width = legacy_upper - legacy_lower
    updated_width = prediction.upper - prediction.lower
    inclusion_violation = np.maximum(
        prediction.lower - legacy_lower,
        legacy_upper - prediction.upper,
    )
    return {
        "dataset": name,
        "certificate": asdict(model.certificate),
        "approved": model.approved,
        "authority_set": [model.authority_lower, model.authority_upper],
        "calibration": {
            "alpha": model.alpha,
            "residual_scale": model.residual_scale,
            "block_quantile": model.block_quantile,
            "n_calibration_units": model.n_calibration_units,
        },
        "point": {
            "mean_unit_log_rmse_improvement": float(np.mean(effect)),
            "unit_wins": int(np.sum(effect > 0)),
            "n_units": len(effect),
            "always_pp_mean_unit_log_rmse_improvement": float(
                np.mean(always_pp_effect)
            ),
        },
        "interval": {
            "current_coverage": float(np.mean(legacy_covered)),
            "updated_coverage": float(np.mean(updated_covered)),
            "coverage_gain": float(
                np.mean(updated_covered) - np.mean(legacy_covered)
            ),
            "current_mean_width": float(np.mean(legacy_width)),
            "updated_mean_width": float(np.mean(updated_width)),
            "mean_width_ratio": float(
                np.mean(updated_width) / max(np.mean(legacy_width), 1e-12)
            ),
            "max_inclusion_violation": float(
                max(0.0, np.max(inclusion_violation))
            ),
            "rows": len(y),
            "current_covered_rows": int(np.sum(legacy_covered)),
            "updated_covered_rows": int(np.sum(updated_covered)),
        },
        "arrays": {
            "y": y,
            "point": prediction.point,
            "lower": prediction.lower,
            "upper": prediction.upper,
            "legacy_lower": legacy_lower,
            "legacy_upper": legacy_upper,
        },
    }


def main():
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite FCC-PPX experiment")
    metadata = json.loads((SOURCE / "results.json").read_text())
    names = list(metadata["datasets"])
    rows = [load_domain(name) for name in names]

    point_effect = np.asarray([
        row["point"]["mean_unit_log_rmse_improvement"] for row in rows
    ])
    active = np.asarray([row["approved"] for row in rows], dtype=bool)
    rng = np.random.default_rng(20260913)
    index = rng.integers(0, len(rows), size=(50_000, len(rows)))
    bootstrap = np.mean(point_effect[index], axis=1)
    current_coverage = np.asarray([
        row["interval"]["current_coverage"] for row in rows
    ])
    updated_coverage = np.asarray([
        row["interval"]["updated_coverage"] for row in rows
    ])
    current_rows = sum(
        row["interval"]["current_covered_rows"] for row in rows
    )
    updated_rows = sum(
        row["interval"]["updated_covered_rows"] for row in rows
    )
    total_rows = sum(row["interval"]["rows"] for row in rows)
    width_ratio = np.asarray([
        row["interval"]["mean_width_ratio"] for row in rows
    ])
    summary = {
        "domains": len(rows),
        "approved_pp_domains": int(np.sum(active)),
        "fallback_domains": int(np.sum(~active)),
        "point_equal_domain_mean_unit_log_rmse_improvement": float(
            np.mean(point_effect)
        ),
        "point_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "point_improved_domains": int(np.sum(point_effect > 1e-12)),
        "point_harmed_domains": int(np.sum(point_effect < -1e-12)),
        "current_equal_domain_coverage": float(np.mean(current_coverage)),
        "updated_equal_domain_coverage": float(np.mean(updated_coverage)),
        "current_pooled_row_coverage": float(current_rows / total_rows),
        "updated_pooled_row_coverage": float(updated_rows / total_rows),
        "noninferior_coverage_domains": int(np.sum(
            updated_coverage >= current_coverage - 1e-15
        )),
        "strictly_improved_coverage_domains": int(np.sum(
            updated_coverage > current_coverage + 1e-15
        )),
        "mean_domain_width_ratio": float(np.mean(width_ratio)),
        "maximum_interval_inclusion_violation": float(max(
            row["interval"]["max_inclusion_violation"] for row in rows
        )),
    }
    payload_rows = []
    arrays = {}
    for row in rows:
        values = row.pop("arrays")
        payload_rows.append(row)
        for key, value in values.items():
            arrays[f"{row['dataset']}_{key}"] = value
    payload = {
        "model": "Falsification-Calibrated Credal Conformal PP-X (FCC-PPX)",
        "status": "opened retrospective structural development; not confirmation",
        "owner": "박진서",
        "protocol": PROTOCOL,
        "summary": summary,
        "datasets": payload_rows,
        "guardrails": [
            "No test outcome enters fitting, certification, or interval construction.",
            "The legacy PP interval is nested row by row.",
            "Point fallback is exact whenever the PP candidate is falsified.",
            "No threshold or width parameter was selected on test outcomes.",
            "All width inflation is reported as the cost of coverage preservation.",
            "An unopened cohort is required for confirmatory promotion.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
