#!/usr/bin/env python3
"""Fair same-information plain-MLP ablation for PP late-tail experiments."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.model import (
    equal_group_weights,
    fit_feature_scale,
    fit_pp,
    predict,
    select_affine_initialization,
    transform_features,
)


class PlainMLP(nn.Module):
    """PP nonlinear path trained as a standalone direct RUL predictor."""

    def __init__(self, input_dim: int, width: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).squeeze(1)


def fit_plain(
    train: dict,
    validation: dict,
    *,
    seed: int,
    max_epochs: int = 300,
    patience: int = 70,
    batch_size: int = 512,
) -> dict:
    torch.manual_seed(seed)
    center, scale = fit_feature_scale(train["x"])
    target_scale = max(float(np.max(train["y"])), 1.0)
    x = torch.as_tensor(
        transform_features(train["x"], center, scale), dtype=torch.float32
    )
    y = torch.as_tensor(train["y"] / target_scale, dtype=torch.float32)
    weights = torch.as_tensor(
        equal_group_weights(train["groups"]), dtype=torch.float32
    )
    validation_x = torch.as_tensor(
        transform_features(validation["x"], center, scale), dtype=torch.float32
    )
    validation_y = np.asarray(validation["y"], dtype=np.float64)
    model = PlainMLP(x.shape[1], width=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2.0)
    rng = np.random.default_rng(seed)

    def validation_mse() -> float:
        model.eval()
        with torch.no_grad():
            estimate = np.clip(
                model(validation_x).cpu().numpy() * target_scale,
                0.0,
                target_scale,
            )
        return float(np.mean((estimate - validation_y) ** 2))

    best_loss = validation_mse()
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    last_epoch = 0
    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            index = torch.as_tensor(order[start : start + batch_size])
            loss = torch.mean(weights[index] * (model(x[index]) - y[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        current = validation_mse()
        if current < best_loss - 1e-10:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch > patience:
            break
    model.load_state_dict(best_state)
    return {
        "model": model,
        "center": center,
        "scale": scale,
        "target_scale": target_scale,
        "selected_epoch": best_epoch,
        "epochs_executed": last_epoch,
        "validation_mse": best_loss,
    }


def predict_plain(fit: dict, x: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(
        transform_features(x, fit["center"], fit["scale"]), dtype=torch.float32
    )
    fit["model"].eval()
    with torch.no_grad():
        estimate = fit["model"](value).cpu().numpy() * fit["target_scale"]
    return np.clip(estimate, 0.0, fit["target_scale"])


def summarize(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key]["pooled"]["r2"] for row in rows])
    macros = np.asarray([row[key]["unit_macro_r2"] for row in rows])
    return {
        "pooled_r2_mean": float(values.mean()),
        "pooled_r2_sd": float(values.std(ddof=0)),
        "unit_macro_r2_mean": float(macros.mean()),
        "unit_macro_r2_sd": float(macros.std(ddof=0)),
    }


def run_single(train, validation, test, seeds) -> dict:
    affine_selection = select_affine_initialization(train, validation)
    rows = []
    for seed in seeds:
        plain_fit = fit_plain(train, validation, seed=seed)
        pp_fit = fit_pp(
            train, validation, seed=seed, affine_selection=affine_selection
        )
        plain_prediction = predict_plain(plain_fit, test["x"])
        pp_prediction = predict(pp_fit, test["x"])
        row = {
            "seed": seed,
            "plain_selected_epoch": plain_fit["selected_epoch"],
            "pp_selected_epoch": pp_fit.selection["selected_epoch"],
            "plain_metrics": regression_metrics(
                test["y"], plain_prediction, test["groups"]
            ),
            "pp_metrics": regression_metrics(test["y"], pp_prediction, test["groups"]),
        }
        rows.append(row)
        print(
            f"seed={seed} plain={row['plain_metrics']['pooled']['r2']:.4f} "
            f"PP={row['pp_metrics']['pooled']['r2']:.4f}",
            flush=True,
        )
    return {
        "plain_mlp": summarize(rows, "plain_metrics"),
        "pp": summarize(rows, "pp_metrics"),
        "runs": rows,
    }


def run_nasa(folds, seeds) -> dict:
    selections = [
        select_affine_initialization(fold["train"], fold["validation"])
        for fold in folds
    ]
    truth = np.concatenate([fold["test"]["y"] for fold in folds])
    groups = np.concatenate([fold["test"]["groups"] for fold in folds])
    rows = []
    for seed in seeds:
        plain_parts, pp_parts, epochs = [], [], []
        for fold, selection in zip(folds, selections):
            plain_fit = fit_plain(fold["train"], fold["validation"], seed=seed)
            pp_fit = fit_pp(
                fold["train"], fold["validation"], seed=seed,
                affine_selection=selection,
            )
            plain_parts.append(predict_plain(plain_fit, fold["test"]["x"]))
            pp_parts.append(predict(pp_fit, fold["test"]["x"]))
            epochs.append({
                "test_cell": fold["test_cell"],
                "plain": plain_fit["selected_epoch"],
                "pp": pp_fit.selection["selected_epoch"],
            })
        plain_prediction = np.concatenate(plain_parts)
        pp_prediction = np.concatenate(pp_parts)
        row = {
            "seed": seed,
            "fold_epochs": epochs,
            "plain_metrics": regression_metrics(truth, plain_prediction, groups),
            "pp_metrics": regression_metrics(truth, pp_prediction, groups),
        }
        rows.append(row)
        print(
            f"NASA seed={seed} plain={row['plain_metrics']['pooled']['r2']:.4f} "
            f"PP={row['pp_metrics']['pooled']['r2']:.4f}",
            flush=True,
        )
    return {
        "plain_mlp": summarize(rows, "plain_metrics"),
        "pp": summarize(rows, "pp_metrics"),
        "runs": rows,
    }


def write_report(payload: dict, path: Path) -> None:
    datasets = payload["datasets"]
    lines = [
        "# PP affine-tail path ablation", "",
        "Plain MLP and PP use identical causal inputs, unit-disjoint strict-tail splits, "
        "two width-32 tanh layers, optimizer, seed set, group weighting, clipping, and "
        "validation checkpoint rule. Plain MLP uses standard PyTorch initialization; "
        "PP adds its frozen validation-selected affine path and zero-start correction.", "",
        "| dataset | plain MLP pooled R² | PP pooled R² | PP gain |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("hust", "HUST"), ("virkler", "Virkler"), ("nasa", "NASA")):
        plain = datasets[key]["plain_mlp"]
        pp = datasets[key]["pp"]
        lines.append(
            f"| {label} | {plain['pooled_r2_mean']:.3f}±{plain['pooled_r2_sd']:.3f} | "
            f"**{pp['pooled_r2_mean']:.3f}±{pp['pooled_r2_sd']:.3f}** | "
            f"{pp['pooled_r2_mean'] - plain['pooled_r2_mean']:+.3f} |"
        )
    lines += [
        "", "This is a retrospective architectural ablation on previously inspected "
        "datasets. It tests the affine-tail path against a matched plain NN; it is not "
        "independent confirmation of a fallback routing rule.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output", default="results/plain_mlp_ablation_v1")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.legacy_root).expanduser().resolve()))
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from run_affine_tail_external_three import prepare_hust, prepare_virkler

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    torch.set_num_threads(2)
    hust = prepare_hust()
    virkler = prepare_virkler()
    nasa_folds, _ = prepare_folds()
    payload = {
        "experiment": "plain_mlp_ablation_v1",
        "status": "retrospective same-information architecture ablation",
        "seeds": list(seeds),
        "datasets": {
            "hust": run_single(*hust[:3], seeds),
            "virkler": run_single(*virkler[:3], seeds),
            "nasa": run_nasa(nasa_folds, seeds),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(payload, output / "RESULTS.md")
    print(f"saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
