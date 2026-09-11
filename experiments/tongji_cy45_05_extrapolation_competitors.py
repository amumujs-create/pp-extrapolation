#!/usr/bin/env python3
"""Figure-matched competitors on Tongji CY45-05 vs sealed CCMR v2.0."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT / ".benchmark_deps"),
]

import tongji_cy45_05_ccmr_v20 as tongji
import two_success_cohorts_extrapolation_competitors as base

OUT = ROOT / "results/tongji_cy45_05_extrapolation_competitors"
SEALED = ROOT / "results/tongji_cy45_05_ccmr_v20/sealed_predictions.npz"


def build_parts():
    trajectories = tongji.load_trajectories()
    split = tongji.split_units(trajectories)
    train = tongji.make_rows(
        trajectories, split["train"], tongji.TRAIN_INTERVAL
    )
    validation = tongji.make_rows(trajectories, split["validation"])
    test = tongji.make_rows(trajectories, split["test"])
    validation_choose = tongji.select(validation, tongji.VAL_INTERVAL)
    test_choose = tongji.select(test, tongji.TEST_INTERVAL)

    def pack(rows, choose=None):
        if choose is None:
            choose = np.ones(len(rows["y"]), dtype=bool)
        return {
            "x": np.asarray(rows["context"][choose], dtype=float),
            "y": np.asarray(rows["y"][choose], dtype=float),
            "groups": np.asarray(rows["groups"][choose]),
        }

    return (
        pack(train),
        pack(validation, validation_choose),
        pack(test, test_choose),
        {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
        },
    )


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite Tongji competitor benchmark")
    train, validation, test, split = build_parts()
    parts = (train, validation, test)
    sealed = np.load(SEALED)
    if not np.allclose(sealed["truth"], test["y"]):
        raise RuntimeError("sealed truth misalignment")
    if not np.array_equal(
        sealed["groups"].astype(str), test["groups"].astype(str)
    ):
        raise RuntimeError("sealed groups misalignment")
    if not np.allclose(sealed["persistence"], test["x"][:, 0]):
        raise RuntimeError("sealed persistence misalignment")

    cohort_result = {
        "rows": {
            "train": len(train["y"]),
            "validation": len(validation["y"]),
            "test": len(test["y"]),
        },
        "units": {
            "train": len(np.unique(train["groups"])),
            "validation": len(np.unique(validation["groups"])),
            "test": len(np.unique(test["groups"])),
        },
        "split": split,
        "models": {
            "persistence": {
                "ensemble": base.score(test, test["x"][:, 0])
            },
            "CCMR_v2.0": {
                "version": "CCMR_v2.0",
                "source": str(SEALED.relative_to(ROOT)),
                "ensemble": base.score(test, sealed["deployed"]),
            },
        },
    }
    prediction_artifacts = {
        "truth": test["y"],
        "groups": test["groups"].astype(str),
        "persistence": test["x"][:, 0],
        "CCMR_v2.0": sealed["deployed"],
    }
    result = {
        "status": "retrospective opened-test competitor benchmark",
        "confirmatory": False,
        "cohort": "Tongji_CY45-05",
        "selection": "validation only",
        "output_contract": (
            "nonnegative predictions; no train-range upper clipping"
        ),
        "seeds": list(base.SEEDS),
        "models": [
            "CCMR_v2.0",
            "V-REx",
            "GroupDRO",
            "Monotone NN",
            "Linear-tail RBF",
            "Engression",
            "Linear-mean GP",
            "TabPFN v3",
        ],
        "cohorts": {"Tongji_CY45-05": cohort_result},
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    for kind, label in (
        ("vrex", "V-REx"),
        ("groupdro", "GroupDRO"),
        ("monotone", "Monotone_NN"),
    ):
        started = time.monotonic()
        model_result, prediction = base.tune_neural(kind, parts)
        model_result["total_seconds"] = time.monotonic() - started
        cohort_result["models"][label] = model_result
        prediction_artifacts[label] = prediction
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(label, flush=True)

    for label, runner in (
        ("Linear_tail_RBF", base.linear_rff),
        ("Engression", base.run_engression),
        ("Linear_mean_GP", base.run_gp),
        ("TabPFN_v3", base.run_tabpfn),
    ):
        started = time.monotonic()
        model_result, prediction = runner(parts)
        model_result["total_seconds"] = time.monotonic() - started
        cohort_result["models"][label] = model_result
        prediction_artifacts[label] = prediction
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(label, flush=True)

    np.savez_compressed(
        OUT / "tongji_cy45_05_ensemble_predictions.npz",
        **prediction_artifacts,
    )
    scores = {
        name: payload["ensemble"]
        for name, payload in cohort_result["models"].items()
        if "ensemble" in payload
    }
    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1]["pooled"]["r2"],
            item[1]["raw_regret"]["maximum"],
            item[1]["pooled"]["rmse"],
        ),
    )
    result["rank_by_pooled_r2_then_max_regret"] = [
        {
            "model": name,
            "pooled_r2": metrics["pooled"]["r2"],
            "pooled_rmse": metrics["pooled"]["rmse"],
            "pooled_rmse_improvement": metrics["pooled_rmse_improvement"],
            "macro_rmse_improvement": metrics["macro_rmse_improvement"],
            "raw_regret_max": metrics["raw_regret"]["maximum"],
        }
        for name, metrics in ranked
    ]
    result["accuracy_leader"] = ranked[0][0]
    ccmr = scores["CCMR_v2.0"]
    engression = scores.get("Engression")
    if engression is not None:
        result["ccmr_beats_engression_on_rmse_and_max_regret"] = bool(
            ccmr["pooled"]["rmse"] < engression["pooled"]["rmse"]
            and ccmr["raw_regret"]["maximum"]
            <= engression["raw_regret"]["maximum"]
        )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "accuracy_leader": result["accuracy_leader"],
        "rank_head": result["rank_by_pooled_r2_then_max_regret"][:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
