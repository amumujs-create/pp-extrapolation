#!/usr/bin/env python3
"""Support-gated PP on strict late-tail HUST, Virkler, and NASA splits."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.model import fit_pp, predict, select_affine_initialization
from pp_extrapolation.support_gate import (
    predict_components,
    predict_support_gated,
    select_support_gate,
    support_distance,
)


BETAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=0))}


def summarize(runs: list[dict], key: str) -> dict:
    return {
        "pooled_r2": mean_sd([row[key]["pooled"]["r2"] for row in runs]),
        "pooled_rmse": mean_sd([row[key]["pooled"]["rmse"] for row in runs]),
        "unit_macro_r2": mean_sd([row[key]["unit_macro_r2"] for row in runs]),
    }


def ensemble_metrics(
    truth: np.ndarray,
    groups: np.ndarray,
    prediction_file: dict,
    prefix: str,
) -> dict:
    prediction = np.mean(
        [value for key, value in prediction_file.items() if key.startswith(prefix)],
        axis=0,
    )
    return regression_metrics(truth, prediction, groups)


def run_single(
    name: str,
    train: dict,
    validation: dict,
    test: dict,
    seeds: tuple[int, ...],
    max_epochs: int,
) -> tuple[dict, dict]:
    validation_distance, validation_audit = support_distance(
        train["x"][:, [0]], validation["x"][:, [0]]
    )
    test_distance, test_audit = support_distance(
        train["x"][:, [0]], test["x"][:, [0]]
    )
    if validation_audit.outside_fraction < 1.0 or test_audit.outside_fraction < 1.0:
        raise RuntimeError(f"{name}: strict tail hull gate failed")
    affine_selection = select_affine_initialization(train, validation)
    runs, prediction_file = [], {"truth": test["y"], "groups": test["groups"]}
    for seed in seeds:
        fit = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=affine_selection,
            max_epochs=max_epochs,
        )
        baseline = predict(fit, test["x"])
        gate = select_support_gate(
            fit,
            validation["x"],
            validation["y"],
            validation_distance,
            betas=BETAS,
        )
        gated = predict_support_gated(
            fit, test["x"], test_distance, beta=gate.beta
        )
        affine, _ = predict_components(fit, test["x"])
        affine = np.clip(affine, 0.0, fit.target_scale)
        row = {
            "seed": seed,
            "selected_beta": gate.beta,
            "beta_candidates": list(gate.candidates),
            "checkpoint": fit.selection,
            "affine_metrics": regression_metrics(test["y"], affine, test["groups"]),
            "pp_metrics": regression_metrics(test["y"], baseline, test["groups"]),
            "gated_metrics": regression_metrics(test["y"], gated, test["groups"]),
        }
        runs.append(row)
        prediction_file[f"affine_seed{seed}"] = affine
        prediction_file[f"pp_seed{seed}"] = baseline
        prediction_file[f"gated_seed{seed}"] = gated
        print(
            f"{name} seed={seed} beta={gate.beta:g} "
            f"PP={row['pp_metrics']['pooled']['r2']:.4f} "
            f"SG-PP={row['gated_metrics']['pooled']['r2']:.4f}",
            flush=True,
        )
    result = {
        "validation_hull": validation_audit.summary(),
        "test_hull": test_audit.summary(),
        "affine": summarize(runs, "affine_metrics"),
        "pp": summarize(runs, "pp_metrics"),
        "support_gated_pp": summarize(runs, "gated_metrics"),
        "ensembles": {
            "pp": ensemble_metrics(test["y"], test["groups"], prediction_file, "pp_seed"),
            "support_gated_pp": ensemble_metrics(
                test["y"], test["groups"], prediction_file, "gated_seed"
            ),
        },
        "runs": runs,
    }
    return result, prediction_file


def run_nasa(folds, seeds: tuple[int, ...], max_epochs: int) -> tuple[dict, dict]:
    selections = [
        select_affine_initialization(fold["train"], fold["validation"])
        for fold in folds
    ]
    truth = np.concatenate([fold["test"]["y"] for fold in folds])
    groups = np.concatenate([fold["test"]["groups"] for fold in folds])
    output = {"truth": truth, "groups": groups}
    runs = []
    fold_audits = []
    for fold in folds:
        _, validation_audit = support_distance(
            fold["train"]["x"][:, [0]], fold["validation"]["x"][:, [0]]
        )
        _, test_audit = support_distance(
            fold["train"]["x"][:, [0]], fold["test"]["x"][:, [0]]
        )
        fold_audits.append({
            "test_cell": fold["test_cell"],
            "validation": validation_audit.summary(),
            "test": test_audit.summary(),
        })
    for seed in seeds:
        affine_parts, pp_parts, gated_parts, betas = [], [], [], []
        for fold, selection in zip(folds, selections):
            validation_distance, _ = support_distance(
                fold["train"]["x"][:, [0]], fold["validation"]["x"][:, [0]]
            )
            test_distance, _ = support_distance(
                fold["train"]["x"][:, [0]], fold["test"]["x"][:, [0]]
            )
            fit = fit_pp(
                fold["train"],
                fold["validation"],
                seed=seed,
                affine_selection=selection,
                max_epochs=max_epochs,
            )
            gate = select_support_gate(
                fit,
                fold["validation"]["x"],
                fold["validation"]["y"],
                validation_distance,
                betas=BETAS,
            )
            affine, _ = predict_components(fit, fold["test"]["x"])
            affine_parts.append(np.clip(affine, 0.0, fit.target_scale))
            pp_parts.append(predict(fit, fold["test"]["x"]))
            gated_parts.append(predict_support_gated(
                fit, fold["test"]["x"], test_distance, beta=gate.beta
            ))
            betas.append({"test_cell": fold["test_cell"], "beta": gate.beta})
        affine = np.concatenate(affine_parts)
        baseline = np.concatenate(pp_parts)
        gated = np.concatenate(gated_parts)
        row = {
            "seed": seed,
            "selected_betas": betas,
            "affine_metrics": regression_metrics(truth, affine, groups),
            "pp_metrics": regression_metrics(truth, baseline, groups),
            "gated_metrics": regression_metrics(truth, gated, groups),
        }
        runs.append(row)
        output[f"affine_seed{seed}"] = affine
        output[f"pp_seed{seed}"] = baseline
        output[f"gated_seed{seed}"] = gated
        print(
            f"NASA seed={seed} betas={[x['beta'] for x in betas]} "
            f"PP={row['pp_metrics']['pooled']['r2']:.4f} "
            f"SG-PP={row['gated_metrics']['pooled']['r2']:.4f}",
            flush=True,
        )
    result = {
        "fold_hulls": fold_audits,
        "affine": summarize(runs, "affine_metrics"),
        "pp": summarize(runs, "pp_metrics"),
        "support_gated_pp": summarize(runs, "gated_metrics"),
        "ensembles": {
            "pp": ensemble_metrics(truth, groups, output, "pp_seed"),
            "support_gated_pp": ensemble_metrics(truth, groups, output, "gated_seed"),
        },
        "runs": runs,
    }
    return result, output


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Support-gated PP: strict late-tail extrapolation", "",
        "Train, validation, and final-test physical units are disjoint. Validation and "
        "test rows are 100% beyond the one-dimensional train hull along the declared "
        "degradation coordinate. Beta and neural checkpoints use validation only.", "",
        "The model is `affine(x) + exp(-beta * hull_distance) * correction_NN(x)`. "
        "Beta zero exactly recovers PP; far outside support the correction vanishes.", "",
        "| dataset | affine/Ridge head pooled R² | PP pooled R² | support-gated PP pooled R² | 5-seed PP ensemble |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("hust", "HUST"), ("virkler", "Virkler"), ("nasa", "NASA battery")):
        row = payload["datasets"][key]
        affine = row["affine"]["pooled_r2"]
        pp = row["pp"]["pooled_r2"]
        gated = row["support_gated_pp"]["pooled_r2"]
        lines.append(
            f"| {label} | {affine['mean']:.3f}±{affine['sd']:.3f} | "
            f"{pp['mean']:.3f}±{pp['sd']:.3f} | "
            f"{gated['mean']:.3f}±{gated['sd']:.3f} "
            f"({gated['mean'] - pp['mean']:+.3f}) | "
            f"**{row['ensembles']['pp']['pooled']['r2']:.3f}** |"
        )
    lines += [
        "", "The primary metric is raw pooled R². HUST uses a censored proxy endpoint; "
        "Virkler and NASA use observed endpoints. The support gate is an ablation, not "
        "the default: it helped HUST seed-average performance, was neutral on Virkler, "
        "and was slightly worse on NASA. These are development results because the same "
        "source datasets were used in earlier architecture work.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--output", default="results/support_gated_cross_domain_v1")
    args = parser.parse_args()
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    sys.path.insert(0, str(legacy_root))
    from run_affine_tail_external_three import (  # pylint: disable=import-outside-toplevel
        prepare_hust,
        prepare_virkler,
    )
    from run_affine_tail_external_nasa_health_v2 import (  # pylint: disable=import-outside-toplevel
        prepare_folds as prepare_nasa_folds,
    )

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()

    hust_train, hust_validation, hust_test, hust_meta = prepare_hust()
    virkler_train, virkler_validation, virkler_test, virkler_meta = prepare_virkler()
    nasa_folds, nasa_meta = prepare_nasa_folds()
    hust, hust_predictions = run_single(
        "HUST", hust_train, hust_validation, hust_test, seeds, args.max_epochs
    )
    virkler, virkler_predictions = run_single(
        "Virkler", virkler_train, virkler_validation, virkler_test, seeds, args.max_epochs
    )
    nasa, nasa_predictions = run_nasa(nasa_folds, seeds, args.max_epochs)
    np.savez_compressed(output / "predictions_hust.npz", **hust_predictions)
    np.savez_compressed(output / "predictions_virkler.npz", **virkler_predictions)
    np.savez_compressed(output / "predictions_nasa.npz", **nasa_predictions)
    payload = {
        "experiment": "support_gated_cross_domain_v1",
        "primary_metric": "raw_pooled_r2",
        "selection": "unit-disjoint hull-out validation only",
        "beta_candidates": list(BETAS),
        "seeds": list(seeds),
        "datasets": {"hust": hust, "virkler": virkler, "nasa": nasa},
        "data_audits": {
            "hust": hust_meta["audit"],
            "virkler": virkler_meta["audit"],
            "nasa": nasa_meta,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(payload, output / "RESULTS.md")
    print(f"saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
