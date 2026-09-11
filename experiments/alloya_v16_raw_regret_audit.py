#!/usr/bin/env python3
"""Audit frozen Alloy A predictions against corrected raw-regret selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import alloya_unopened_ccmr_v16 as alloy
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

OUT = ROOT / "results/alloya_unopened_ccmr_v16/raw_regret_audit.json"


def main():
    if OUT.exists():
        raise RuntimeError("refusing to overwrite raw-regret audit")
    frame = next(iter(pyreadr.read_r(str(alloy.DATA)).values()))
    frame.columns = [str(column).lower() for column in frame.columns]
    labels = sorted(frame["specimen"].astype(str).unique())
    split = alloy.frozen_split(labels)
    rows = {
        mode: alloy.make_rows(frame, split[mode], mode)
        for mode in ("train", "validation", "test")
    }
    train, validation, test = (
        rows["train"], rows["validation"], rows["test"]
    )
    model = fit_consensus_residual(
        train["correction"],
        train["context"],
        train["y"],
        train["groups"],
        train["context"][:, 0],
        validation["correction"],
        validation["context"],
        validation["y"],
        validation["groups"],
        validation["context"][:, 0],
    )
    validation_prediction, _ = predict_consensus_residual(
        model,
        validation["correction"],
        validation["context"],
        validation["context"][:, 0],
    )
    test_prediction, evidence = predict_consensus_residual(
        model,
        test["correction"],
        test["context"],
        test["context"][:, 0],
    )
    sealed = np.load(
        ROOT / "results/alloya_unopened_ccmr_v16/sealed_predictions.npz"
    )
    maximum_difference = float(np.max(np.abs(
        test_prediction - sealed["prediction"]
    )))
    validation_regret = regret_summary(raw_unit_regret(
        validation["y"],
        validation["groups"],
        validation["context"][:, 0],
        validation_prediction,
    ))
    test_regret = regret_summary(raw_unit_regret(
        test["y"],
        test["groups"],
        test["context"][:, 0],
        test_prediction,
    ))
    payload = {
        "status": "post-score metric-definition audit",
        "issue": (
            "the original selector used stabilized relative unit regret while "
            "the frozen protocol specified raw relative excess MSE"
        ),
        "corrected_selector_deployment_mass": model.deployment_mass,
        "sealed_prediction_maximum_absolute_difference": maximum_difference,
        "active_mask_identical": bool(np.array_equal(
            evidence["active"], sealed["active"]
        )),
        "validation_raw_regret": {
            "mean": validation_regret[0],
            "cvar20": validation_regret[1],
            "maximum": validation_regret[2],
        },
        "test_raw_regret": {
            "mean": test_regret[0],
            "cvar20": test_regret[1],
            "maximum": test_regret[2],
        },
        "protocol_result_preserved": bool(
            maximum_difference == 0
            and validation_regret[0] <= 0
            and validation_regret[1] <= 0.01
            and validation_regret[2] <= 0.02
            and test_regret[0] <= 0
            and test_regret[1] <= 0.02
            and test_regret[2] <= 0.05
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
