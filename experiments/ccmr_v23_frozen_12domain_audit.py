#!/usr/bin/env python3
"""Frozen, no-refit 12-domain compatibility audit for CCMR v2.3."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

OUT = ROOT / "results/ccmr_v23_frozen_12domain_audit"
REGISTRY = ROOT / "results/ppx_unified_final_v1/registry.json"
PAIRED = ROOT / "results/final_modular_pp_evidence_v1/results.json"
COMPETITORS = (
    ROOT / "results/extrapolation_competitors_all_v1/results.json"
)
KEYS = {
    "HUST": "hust",
    "Virkler": "virkler",
    "NASA battery": "nasa",
    "Sunwoda": "sunwoda",
    "RWTH": "rwth",
    "MATR2019": "matr",
    "MATR batch 2": "matr_batch2",
    "N-CMAPSS": "ncmapss",
    "MICH": "mich",
    "XJTU": "xjtu",
    "FEMTO": "femto",
    "NASA milling": "milling",
}
CONTRACTS = {
    "HUST": "condition_transfer",
    "Virkler": "known_boundary",
    "NASA battery": "latent_rul",
    "Sunwoda": "known_boundary",
    "RWTH": "known_boundary",
    "MATR2019": "latent_rul",
    "MATR batch 2": "condition_transfer",
    "N-CMAPSS": "condition_transfer",
    "MICH": "known_boundary",
    "XJTU": "condition_transfer",
    "FEMTO": "latent_rul",
    "NASA milling": "known_boundary",
}
RETROSPECTIVE = {"XJTU", "FEMTO", "NASA milling"}


def _ensemble_r2(model):
    try:
        return float(model["ensemble"]["pooled"]["r2"])
    except (KeyError, TypeError):
        return None


def _best_competitor(dataset):
    candidates = []
    for model, result in dataset.items():
        value = _ensemble_r2(result)
        if value is not None and np.isfinite(value):
            candidates.append((value, model))
    if not candidates:
        return {"model": None, "pooled_r2": None}
    value, model = max(candidates)
    return {"model": model, "pooled_r2": value}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite frozen 12-domain audit")
    registry = json.loads(REGISTRY.read_text())
    paired = json.loads(PAIRED.read_text())
    competitors = json.loads(COMPETITORS.read_text())["datasets"]
    paired_by_name = {
        item["dataset"].replace("SUNWODA", "Sunwoda")
        .replace("MATR-b2", "MATR batch 2")
        .replace("NASA", "NASA battery"): item
        for item in paired["datasets"]
    }
    # Correct replacements that should retain their original names.
    for item in paired["datasets"]:
        if item["dataset"] in ("N-CMAPSS", "MICH", "RWTH", "Virkler", "HUST"):
            paired_by_name[item["dataset"]] = item
    rows = []
    missing = []
    for route in registry["routes"]:
        name = route["dataset"]
        artifact = ROOT / route["artifact"]
        if not artifact.exists():
            missing.append(str(artifact.relative_to(ROOT)))
        baseline = _best_competitor(competitors.get(KEYS[name], {}))
        paired_item = paired_by_name.get(name)
        rows.append({
            "dataset": name,
            "contract": CONTRACTS[name],
            "tier": (
                "retrospective_extension"
                if name in RETROSPECTIVE else "paired_final"
            ),
            "final_pp_r2": route["development_pooled_r2"],
            "route": route["route"],
            "result_artifact_exists": artifact.exists(),
            "best_common_competitor": baseline,
            "paired_row_alignment": paired_item is not None,
            "paired_baseline": (
                paired_item["baseline"] if paired_item else None
            ),
            "paired_r2_gap": (
                paired_item["r2_gap"] if paired_item else None
            ),
            "development_caveat": (
                "post-test/weak-seed retrospective tier"
                if name in RETROSPECTIVE else None
            ),
        })
    paired_log_effect = [
        float(item["mean_unit_log_rmse_ratio"])
        for item in paired["datasets"]
    ]
    payload = {
        "status": "frozen no-refit retrospective compatibility audit",
        "used_for_v23_selection": False,
        "model_rows": rows,
        "summary": {
            "registered_domains": len(rows),
            "artifacts_present": len(rows) - len(missing),
            "paired_domains": sum(
                row["paired_row_alignment"] for row in rows
            ),
            "retrospective_extension_domains": sum(
                row["tier"] == "retrospective_extension" for row in rows
            ),
            "paired_final_pp_wins": paired["aggregate"][
                "pp_ensemble_wins"
            ],
            "paired_domain_equal_geometric_rmse_reduction": float(
                1.0 - math.exp(-np.mean(paired_log_effect))
            ),
            "missing_result_artifacts": missing,
        },
        "limitations": [
            "The 12 final routes are heterogeneous frozen protocols, not one matched retraining run.",
            "XJTU, FEMTO, and milling remain a separate retrospective tier.",
            "Common competitor JSON lacks row predictions, so only the existing nine-domain paired artifact supports unit-paired inference.",
            "This audit is prohibited from changing CCMR v2.3 settings.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
