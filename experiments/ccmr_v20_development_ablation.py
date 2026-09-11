#!/usr/bin/env python3
"""Descriptive expert ablation after freezing CCMR v2.0."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ccmr_v20_trajectory_development import (
    load_development,
    score,
    selected,
    subset,
)
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)

OUT = ROOT / "results/ccmr_v20_trajectory_development/ablation.json"


def main():
    if OUT.exists():
        raise RuntimeError("refusing to overwrite v2.0 ablation")
    result = {
        "status": "post-freeze descriptive development ablation",
        "used_for_model_selection": False,
        "cohorts": {},
    }
    for name, (train, validation, test, intervals) in (
        load_development().items()
    ):
        validation_choose = selected(validation, intervals[1])
        test_choose = selected(test, intervals[2])
        validation_fit = subset(validation, validation_choose)
        model = fit_causal_dynamics_bank(
            train["correction"],
            train["context"],
            train["y"],
            train["groups"],
            train["context"][:, 0],
            validation_fit["correction"],
            validation_fit["context"],
            validation_fit["y"],
            validation_fit["groups"],
            validation_fit["context"][:, 0],
        )
        variants = {}
        weights = [
            (expert.name, np.eye(len(model.experts))[index])
            for index, expert in enumerate(model.experts)
        ]
        weights.extend([
            ("uniform_bank", np.ones(len(model.experts)) / len(model.experts)),
            ("selected_bank", model.expert_weights),
        ])
        for label, expert_weights in weights:
            variant = replace(
                model,
                expert_weights=np.asarray(expert_weights),
                deployment_mass=1.0,
            )
            prediction, evidence = predict_causal_dynamics_bank(
                variant,
                test["correction"],
                test["context"],
                test["context"][:, 0],
            )
            variants[label] = {
                "base_coverage": float(np.mean(
                    evidence["active"][test_choose]
                )),
                **score(test, prediction, test_choose),
            }
        result["cohorts"][name] = variants
        print(name, flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
