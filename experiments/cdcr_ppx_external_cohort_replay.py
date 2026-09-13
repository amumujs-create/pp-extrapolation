#!/usr/bin/env python3
"""Retrospective CDCR-PPX replay on previously opened external cohorts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from final_modular_pp_evidence import build_datasets, r2  # noqa: E402
from pp_extrapolation.cross_domain_consensus import (  # noqa: E402
    consensus_representation,
    fit_cross_domain_consensus_head,
    normalized_residual_target,
    predict_cross_domain_consensus,
)

OUT = ROOT / "results" / "cdcr_ppx_external_cohort_replay_v1"
PROTOCOL = "protocols/CDCR_PPX_EXTERNAL_COHORT_REPLAY_PROTOCOL.md"


def metric(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "r2": r2(y, prediction),
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "mae": float(np.mean(np.abs(y - prediction))),
    }


def fit_development_head():
    features, targets, domains = [], [], []
    for name, (y, _groups, predictions, *_unused) in build_datasets().items():
        predictions = np.asarray(predictions, dtype=np.float64)
        representation = consensus_representation(predictions)
        features.append(representation.features)
        targets.append(normalized_residual_target(y, representation))
        domains.extend([name] * len(y))
    return fit_cross_domain_consensus_head(
        np.concatenate(features),
        np.concatenate(targets),
        np.asarray(domains),
    )


def external_settings():
    settings = []
    axial = np.load(
        ROOT / "results/axial_fan_three_config_confirmation_v1"
        / "predictions_frozen_before_labels.npz",
        allow_pickle=True,
    )
    for tag in ("1P_8F", "4P_1F", "4P_8F"):
        truth = np.loadtxt(
            ROOT / "data/axial_fan/sealed" / f"RUL_FAN_{tag}.txt"
        ).reshape(-1)
        settings.append({
            "family": "axial_fan",
            "setting": tag,
            "y": truth,
            "pp": np.asarray(axial[f"{tag}_pp"], dtype=np.float64),
            "mlp": np.asarray(axial[f"{tag}_mlp"], dtype=np.float64),
        })
    for family, directory in (
        ("matwi", "matwi_tool_life_untouched_v1"),
        ("misata", "misata_machine_untouched_v1"),
    ):
        saved = np.load(
            ROOT / "results" / directory
            / "predictions_frozen_before_scoring.npz",
            allow_pickle=True,
        )
        settings.append({
            "family": family,
            "setting": family,
            "y": np.asarray(saved["truth"], dtype=np.float64),
            "pp": np.asarray(saved["pp"], dtype=np.float64),
            "mlp": np.asarray(saved["mlp"], dtype=np.float64),
        })
    return settings


def main():
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite external cohort replay")
    model = fit_development_head()
    results = []
    arrays = {}
    for item in external_settings():
        prediction = predict_cross_domain_consensus(model, item["pp"])
        y = item["y"]
        base = np.mean(item["pp"], axis=0)
        mlp = np.mean(item["mlp"], axis=0)
        base_seed_r2 = np.asarray([r2(y, value) for value in item["pp"]])
        updated_seed_r2 = np.asarray([
            r2(y, value) for value in prediction.seeds
        ])
        key = f"{item['family']}_{item['setting']}"
        arrays[f"{key}_y"] = y
        arrays[f"{key}_base_pp"] = item["pp"]
        arrays[f"{key}_updated_pp"] = prediction.seeds
        results.append({
            "family": item["family"],
            "setting": item["setting"],
            "n_rows": len(y),
            "active_meta_residual": prediction.active_residual,
            "q90_seed_disagreement": prediction.q90_seed_disagreement,
            "base_pp": metric(y, base),
            "cdcr_ppx": metric(y, prediction.ensemble),
            "matched_mlp": metric(y, mlp),
            "ensemble_r2_gain_vs_pp": (
                r2(y, prediction.ensemble) - r2(y, base)
            ),
            "base_seed_r2_mean": float(np.mean(base_seed_r2)),
            "updated_seed_r2_mean": float(np.mean(updated_seed_r2)),
            "base_seed_r2_min": float(np.min(base_seed_r2)),
            "updated_seed_r2_min": float(np.min(updated_seed_r2)),
            "base_seed_r2_sd": float(np.std(base_seed_r2, ddof=1)),
            "updated_seed_r2_sd": float(np.std(updated_seed_r2, ddof=1)),
        })

    base_r2 = np.asarray([row["base_pp"]["r2"] for row in results])
    updated_r2 = np.asarray([row["cdcr_ppx"]["r2"] for row in results])
    summary = {
        "external_settings": len(results),
        "base_positive_r2_settings": int(np.sum(base_r2 > 0)),
        "updated_positive_r2_settings": int(np.sum(updated_r2 > 0)),
        "r2_improved_settings": int(np.sum(updated_r2 > base_r2 + 1e-12)),
        "r2_tied_settings": int(np.sum(np.abs(updated_r2 - base_r2) <= 1e-12)),
        "r2_harmed_settings": int(np.sum(updated_r2 < base_r2 - 1e-12)),
        "seed_sd_reduced_settings": int(sum(
            row["updated_seed_r2_sd"] < row["base_seed_r2_sd"]
            for row in results
        )),
        "minimum_setting_r2_base": float(np.min(base_r2)),
        "minimum_setting_r2_updated": float(np.min(updated_r2)),
        "mean_setting_r2_base": float(np.mean(base_r2)),
        "mean_setting_r2_updated": float(np.mean(updated_r2)),
    }
    payload = {
        "model": "Cross-Domain Consensus Residual PP-X (CDCR-PPX)",
        "status": "retrospective replay on previously opened external cohorts",
        "protocol": PROTOCOL,
        "summary": summary,
        "settings": results,
        "head": {
            "ridge_alpha": model.ridge_alpha,
            "training_domains": list(model.training_domains),
            "intercept": model.intercept,
            "coefficient": model.coefficient.tolist(),
        },
        "guardrails": [
            "External labels were not used to fit the head or generate predictions.",
            "All external outcomes were already opened before this replay.",
            "This replay cannot be presented as prospective confirmation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    print(json.dumps(summary, indent=2))
    for row in results:
        print(
            row["family"], row["setting"],
            "PP", round(row["base_pp"]["r2"], 4),
            "CDCR", round(row["cdcr_ppx"]["r2"], 4),
            "MLP", round(row["matched_mlp"]["r2"], 4),
        )


if __name__ == "__main__":
    main()
