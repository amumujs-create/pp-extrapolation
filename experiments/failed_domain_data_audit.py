#!/usr/bin/env python3
"""Causal-data and target-range audit for the two PP failure settings.

No model is fitted and no test target is used to choose a route.  The purpose is
to make the final paper decision reproducible: whether a reported score is a
valid PP comparison, a developmental result, or an information-limited setting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
from femto_sensor_adapter_v2 import OFFICIAL, load as load_femto


def femto_audit() -> dict:
    data = load_femto()
    test = data["role"] == "test"
    endpoints = []
    for bearing in sorted(np.unique(data["bearing"][test])):
        ix = np.flatnonzero(data["bearing"] == bearing)
        endpoint = ix[np.argmax(data["recording_index"][ix])]
        endpoints.append(endpoint)
    endpoint_y = data["y"][endpoints]
    matches = all(abs(float(data["y"][i]) - OFFICIAL[str(data["bearing"][i])]) < 1e-6 for i in endpoints)
    return {
        "adapter": "femto_sensor_adapter_v2",
        "physical_channels": [4, 5],
        "forbidden_directory": "Full_Test_Set",
        "n_learning_bearings": int(len(np.unique(data["bearing"][~test]))),
        "n_test_bearings": int(len(endpoints)),
        "test_rows_per_bearing": 1,
        "endpoint_labels_match_official": matches,
        "test_rul_min_max_seconds": [float(endpoint_y.min()), float(endpoint_y.max())],
        "test_rul_cv": float(endpoint_y.std(ddof=1) / endpoint_y.mean()),
        "paper_status": "endpoint comparison is valid after adapter repair; unit-trajectory R2 and trajectory-level inference are unavailable",
    }


def xjtu_audit() -> dict:
    # Read the already audited feature cache directly.  Importing the training
    # runner would unnecessarily require torch for this read-only audit.
    with np.load(ROOT / "data/xjtu_sy/features_locked.npz", allow_pickle=False) as archive:
        cached = {key: archive[key] for key in archive.files}
    def rul(condition: str) -> np.ndarray:
        mask = cached["conditions"] == condition
        return cached["lives"][mask].astype(float) - cached["positions"][mask].astype(float)
    y_train, y_val, y_test = rul("37.5Hz11kN"), rul("35Hz12kN"), rul("40Hz10kN")
    return {
        "split": "condition-1 train, condition-2 validation, condition-3 test",
        "n_rows": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "rul_range": {
            "train": [float(y_train.min()), float(y_train.max())],
            "validation": [float(y_val.min()), float(y_val.max())],
            "test": [float(y_test.min()), float(y_test.max())],
        },
        "test_rows_above_train_max_fraction": float(np.mean(y_test > y_train.max())),
        "test_to_train_mean_scale": float(y_test.mean() / y_train.mean()),
        "paper_status": "strict operating-condition and target-scale transfer; shared-scale PP is not an adequate final route",
        "available_developmental_route": "validation-reflected scale PP, pooled R2 0.257; it requires a new condition cohort before confirmatory use",
    }


def main() -> None:
    out = ROOT / "results" / "failed_domain_data_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"femto": femto_audit(), "xjtu": xjtu_audit()}
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
