"""Command-line runner for a CSV strict-hull PP experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .hull import audit_convex_hull_support
from .metrics import regression_metrics
from .model import fit_pp, predict, select_affine_initialization
from .split import strict_hull_split_1d


def _model_split(frame: pd.DataFrame, features: list[str], target: str, unit: str) -> dict:
    return {
        "x": frame[features].to_numpy(dtype=np.float32),
        "y": frame[target].to_numpy(dtype=np.float32),
        "groups": frame[unit].astype(str).to_numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PP on a strict 1D hull-out split")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", default="RUL")
    parser.add_argument("--unit", default="unit")
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--train-cutoff", type=float, required=True)
    parser.add_argument("--direction", choices=("low", "high"), required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--output", default="results/pp_run")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    needed = set(args.features + [args.target, args.unit, args.coordinate])
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"missing CSV columns: {missing}")
    if frame[list(needed)].isna().any().any():
        raise ValueError("selected CSV columns contain missing values")

    train_frame, validation_frame, test_frame, split_audit = strict_hull_split_1d(
        frame,
        unit_column=args.unit,
        coordinate=args.coordinate,
        train_cutoff=args.train_cutoff,
        direction=args.direction,
    )
    train = _model_split(train_frame, args.features, args.target, args.unit)
    validation = _model_split(validation_frame, args.features, args.target, args.unit)
    test = _model_split(test_frame, args.features, args.target, args.unit)
    validation_hull = audit_convex_hull_support(
        train_frame[[args.coordinate]].to_numpy(),
        validation_frame[[args.coordinate]].to_numpy(),
        feature_names=[args.coordinate],
    )
    test_hull = audit_convex_hull_support(
        train_frame[[args.coordinate]].to_numpy(),
        test_frame[[args.coordinate]].to_numpy(),
        feature_names=[args.coordinate],
    )
    if validation_hull.outside_fraction < 1.0 or test_hull.outside_fraction < 1.0:
        raise RuntimeError("strict-hull requirement failed")

    torch.set_num_threads(2)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    selection = select_affine_initialization(train, validation)
    predictions = []
    runs = []
    for seed in seeds:
        fit = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=selection,
            max_epochs=args.max_epochs,
        )
        prediction = predict(fit, test["x"])
        predictions.append(prediction)
        runs.append({"selection": fit.selection, "metrics": regression_metrics(test["y"], prediction, test["groups"])})
        print(f"seed={seed} pooled_R2={runs[-1]['metrics']['pooled']['r2']:.6f}", flush=True)

    matrix = np.vstack(predictions)
    r2_values = [run["metrics"]["pooled"]["r2"] for run in runs]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "PP",
        "features": args.features,
        "target": args.target,
        "unit_column": args.unit,
        "hull_coordinate": args.coordinate,
        "split": split_audit,
        "validation_hull": validation_hull.summary(),
        "test_hull": test_hull.summary(),
        "seeds": list(seeds),
        "selected_affine_alpha": float(selection["selected_alpha"]),
        "runs": runs,
        "summary": {
            "pooled_r2_mean": float(np.mean(r2_values)),
            "pooled_r2_sd": float(np.std(r2_values, ddof=0)),
            "ensemble_mean_metrics": regression_metrics(test["y"], matrix.mean(axis=0), test["groups"]),
        },
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "predictions.npz",
        truth=test["y"],
        groups=test["groups"],
        prediction_mean=matrix.mean(axis=0),
        **{f"prediction_seed{seed}": value for seed, value in zip(seeds, matrix)},
    )
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()

