#!/usr/bin/env python3
"""Matched residual-authority ablation: Fixed vs Dual vs 3-scale vs Continuous.

Question
--------
Is dual-scale uniquely necessary, or would a 3-scale / continuous authority
family do as well (or better) under the same residual network and protocol?

Protocol
--------
Same joint battery split and seed schedule as the frozen dual-scale replay:
train → validation epoch select → full refit → source score by dataset.
No test peeking for model choice among the four arms; selection is reported
post-hoc from validation macro MSE (and from the frozen dual gate rule).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT.parent / "ca-css-ncmapss"
sys.path[:0] = [str(ROOT / "src"), str(ADAPTERS), str(ROOT / "experiments")]

from boundary_quotient_pp_batteries import (  # noqa: E402
    build_rows,
    full_part,
    score_by_dataset,
)
from pae_boundary_realdata import DATASETS, prepare_dataset  # noqa: E402
from pae_shared_battery_nn import BatteryRepresentationScale, concatenate_rows  # noqa: E402
from pp_extrapolation import battery_dual_scale_pp_config  # noqa: E402
from pp_extrapolation.boundary_quotient import (  # noqa: E402
    fit_boundary_quotient_pp,
    predict_boundary_quotient,
)

OUT = ROOT / "results/bq_authority_scale_ablation_v1"
SEEDS = (42, 43, 44, 45, 46)
DUAL = battery_dual_scale_pp_config()

# Shared optimizer / width / affine prior with the frozen dual executor.
COMMON = {
    "width": DUAL["width"],
    "alpha": DUAL["alpha"],
    "learning_rate": DUAL["learning_rate"],
    "weight_decay": DUAL["weight_decay"],
}

ARMS = {
    "fixed": {
        **COMMON,
        "residual_bound": 2.0,
    },
    "dual": {
        **COMMON,
        **{k: DUAL[k] for k in (
            "residual_bound",
            "broad_residual_bound",
            "local_saturation_weight",
            "support_gate_feature",
            "support_gate_threshold",
            "support_gate_temperature",
            "support_adaptive_saturation",
        )},
    },
    "tri_scale": {
        **COMMON,
        "residual_bound": 2.0,
        "authority_ladder": (2.0, 4.0, 6.0),
        "support_gate_feature": DUAL["support_gate_feature"],
        "support_gate_threshold": DUAL["support_gate_threshold"],
        "support_gate_temperature": DUAL["support_gate_temperature"],
    },
    "continuous": {
        **COMMON,
        "residual_bound": 2.0,
        "broad_residual_bound": 6.0,
        "continuous_authority": True,
        "support_gate_feature": DUAL["support_gate_feature"],
        "support_gate_threshold": DUAL["support_gate_threshold"],
        "support_gate_temperature": DUAL["support_gate_temperature"],
    },
}


def load_rows():
    scales = {}
    audits = {}
    parts = {key: [] for key in ("train", "validation", "full", "source")}
    for index, name in enumerate(DATASETS):
        split, audit = prepare_dataset(name)
        audits[name] = audit
        scales[name] = BatteryRepresentationScale.fit(split["train"], audit["boundary"])
        for key, part in (
            ("train", split["train"]),
            ("validation", split["val"]),
            ("full", full_part(split)),
            ("source", split["source"]),
        ):
            parts[key].append(build_rows(part, scales[name], index))
    rows = {key: concatenate_rows(value) for key, value in parts.items()}
    return rows, scales, audits


def run_arm(name: str, config: dict, rows: dict, scales: dict) -> dict:
    print(f"ARM {name}", flush=True)
    preds = []
    runs = []
    for seed in SEEDS:
        selected = fit_boundary_quotient_pp(
            rows["train"],
            rows["validation"],
            seed=seed,
            max_epochs=500,
            patience=70,
            **config,
        )
        epochs = max(int(selected.selection["selected_epoch"]), 1)
        fitted = fit_boundary_quotient_pp(
            rows["full"],
            rows["full"],
            seed=seed,
            max_epochs=epochs,
            patience=10000,
            restore_best=False,
            **config,
        )
        prediction = predict_boundary_quotient(fitted, rows["source"])
        preds.append(prediction)
        metrics = score_by_dataset(prediction, rows["source"], scales)
        runs.append(
            {
                "seed": seed,
                "selected_epoch": epochs,
                "validation_mse": selected.selection["validation_dataset_macro_mse"],
                "metrics": metrics,
            }
        )
        print(
            f"  seed={seed} val={runs[-1]['validation_mse']:.4f} "
            f"{ {d: round(metrics[d]['pooled_r2'], 3) for d in DATASETS} }",
            flush=True,
        )
    matrix = np.asarray(preds)
    ensemble = score_by_dataset(matrix.mean(0), rows["source"], scales)
    seed_r2 = np.asarray(
        [[run["metrics"][name]["pooled_r2"] for name in DATASETS] for run in runs]
    )
    mean_val = float(np.mean([run["validation_mse"] for run in runs]))
    return {
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in config.items()
        },
        "runs": runs,
        "ensemble": ensemble,
        "mean_validation_mse": mean_val,
        "dataset_seed_mean": {
            name: float(seed_r2[:, index].mean()) for index, name in enumerate(DATASETS)
        },
        "dataset_seed_sd": {
            name: float(seed_r2[:, index].std(ddof=1)) for index, name in enumerate(DATASETS)
        },
        "min_dataset_r2": float(min(ensemble[name]["pooled_r2"] for name in DATASETS)),
        "mean_dataset_r2": float(np.mean([ensemble[name]["pooled_r2"] for name in DATASETS])),
    }


def dual_gate_vs_fixed(fixed: dict, dual: dict) -> dict:
    """Replay the paper dual-admission rule on this ablation's validation scores."""
    # Approximate support heterogeneity from frozen protocol: MICH-type joint
    # development already passed ≥.50 when dual was admitted. Here we only
    # re-check the ≥2% validation improvement clause on matched seeds.
    fixed_val = fixed["mean_validation_mse"]
    dual_val = dual["mean_validation_mse"]
    admitted = dual_val < fixed_val * 0.98
    return {
        "fixed_mean_validation_mse": fixed_val,
        "dual_mean_validation_mse": dual_val,
        "relative_improvement": 1.0 - dual_val / max(fixed_val, 1e-12),
        "admits_dual_by_2pct_rule": admitted,
    }


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, scales, audits = load_rows()
    arms = {}
    for name, config in ARMS.items():
        arms[name] = run_arm(name, config, rows, scales)

    # Validation-only ranking among the four arms (no test peeking).
    ranking = sorted(arms.items(), key=lambda item: item[1]["mean_validation_mse"])
    selected = ranking[0][0]

    # Paper-style conditional dual vs fixed only.
    gate = dual_gate_vs_fixed(arms["fixed"], arms["dual"])

    payload = {
        "status": "matched residual-authority ablation on joint battery cohorts",
        "question": (
            "Does dual-scale uniquely recover MICH, or do 3-scale / continuous "
            "authority families match or beat it under the same residual net?"
        ),
        "protocol": {
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "fit": "train/val epoch select then full refit; source scored once",
            "matched": "same width/optimizer/affine prior/support feature as dual",
        },
        "arms": arms,
        "validation_selected_arm": selected,
        "validation_ranking": [
            {
                "arm": name,
                "mean_validation_mse": value["mean_validation_mse"],
                "mich_r2": value["ensemble"]["mich"]["pooled_r2"]
                if "mich" in value["ensemble"]
                else value["ensemble"].get(DATASETS[-1], {}).get("pooled_r2"),
                "per_dataset_r2": {
                    dataset: value["ensemble"][dataset]["pooled_r2"] for dataset in DATASETS
                },
            }
            for name, value in ranking
        ],
        "dual_vs_fixed_gate": gate,
        "data_audits": audits,
        "runtime_seconds": time.perf_counter() - started,
    }
    # Fix mich key - DATASETS names may be sunwoda/rwth/mich
    for row in payload["validation_ranking"]:
        per = row["per_dataset_r2"]
        mich_key = next((k for k in per if "mich" in k.lower()), DATASETS[-1])
        row["mich_r2"] = per[mich_key]

    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("SELECTED_BY_VAL", selected)
    print(
        "ENSEMBLE",
        {
            name: {d: round(arms[name]["ensemble"][d]["pooled_r2"], 3) for d in DATASETS}
            for name in ARMS
        },
    )


if __name__ == "__main__":
    main()
