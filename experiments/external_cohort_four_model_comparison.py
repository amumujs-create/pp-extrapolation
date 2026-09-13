#!/usr/bin/env python3
"""Matched external-cohort comparison of MLP, Engression, PP-X, and CDCR-PPX."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / ".benchmark_deps"),
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from engression import engression

from axial_fan_tail_gate_pp_development import tail as axial_tail
from axial_fan_untouched_confirmation import CONFIGS as AXIAL_CONFIGS
from axial_fan_untouched_confirmation import load_features as axial_features
from cdcr_ppx_external_cohort_replay import fit_development_head
from matwi_tool_life_untouched import clean_tools, folds, make_rows
from misata_machine_untouched import prepare as misata_prepare
from misata_machine_untouched import rows as misata_rows
from pp_extrapolation.cross_domain_consensus import predict_cross_domain_consensus

OUT = ROOT / "results" / "external_cohort_four_model_comparison_v1"
PROTOCOL = "protocols/EXTERNAL_COHORT_FOUR_MODEL_COMPARISON_PROTOCOL.md"
SEEDS = tuple(range(42, 47))
GRID = tuple(
    (hidden, learning_rate, beta, layers, epochs)
    for hidden in (32, 64)
    for learning_rate in (1e-3, 5e-3)
    for beta in (0.5, 1.0)
    for layers, epochs in ((2, 250), (3, 500))
)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = y - prediction
    tss = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual**2) / tss),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def tensors(rows: dict):
    return (
        torch.as_tensor(rows["x"], dtype=torch.float32),
        torch.as_tensor(np.asarray(rows["y"])[:, None], dtype=torch.float32),
    )


def train_engression(train: dict, config: tuple, seed: int):
    x, y = tensors(train)
    hidden, learning_rate, beta, layers, epochs = config
    torch.manual_seed(seed)
    return engression(
        x,
        y,
        num_layer=layers,
        hidden_dim=hidden,
        noise_dim=32,
        beta=beta,
        lr=learning_rate,
        num_epoches=epochs,
        batch_size=min(512, len(x)),
        device="cpu",
        standardize=True,
        verbose=False,
    )


def predict_mean(model, rows: dict, sample_size: int) -> np.ndarray:
    x = torch.as_tensor(rows["x"], dtype=torch.float32)
    with torch.no_grad():
        value = model.predict(x, target="mean", sample_size=sample_size)
    return value.squeeze().cpu().numpy().reshape(-1)


def validation_mse(train: dict, validation: dict, config: tuple) -> float:
    model = train_engression(train, config, 42)
    prediction = predict_mean(model, validation, 50)
    cap = max(float(np.max(train["y"])), 1.0)
    prediction = np.clip(prediction, 0.0, cap)
    return float(np.mean((prediction - validation["y"]) ** 2))


def run_engression(
    train: dict,
    validation: dict,
    development: dict,
    test: dict,
    *,
    fold_parts: list[tuple[dict, dict]] | None = None,
) -> tuple[np.ndarray, dict]:
    search = []
    for config in GRID:
        if fold_parts is None:
            score = validation_mse(train, validation, config)
        else:
            score = float(np.mean([
                validation_mse(fold_train, fold_validation, config)
                for fold_train, fold_validation in fold_parts
            ]))
        search.append({"config": list(config), "validation_mse": score})
        print("ENGRESSION_SEARCH", config, score, flush=True)
    selected = min(search, key=lambda row: row["validation_mse"])
    config = tuple(selected["config"])
    cap = max(float(np.max(development["y"])), 1.0)
    predictions = []
    for seed in SEEDS:
        model = train_engression(development, config, seed)
        prediction = np.clip(predict_mean(model, test, 100), 0.0, cap)
        predictions.append(prediction)
        print("ENGRESSION_REFIT", config, seed, flush=True)
    return np.asarray(predictions), {"selected": selected, "search": search}


def axial_settings() -> list[dict]:
    settings = []
    for tag, n_regimes in AXIAL_CONFIGS.items():
        train, validation, development, test, *_ = axial_features(tag, n_regimes)
        train, validation, development = map(axial_tail, (train, validation, development))
        frozen = np.load(
            ROOT / "results" / "axial_fan_tail_gate_pp_development_v1"
            / f"{tag}_predictions.npz",
            allow_pickle=True,
        )
        settings.append({
            "cohort": "Axial-fan",
            "setting": tag,
            "train": train,
            "validation": validation,
            "development": development,
            "test": {**test, "y": np.asarray(frozen["truth"], dtype=np.float64)},
            "ppx": np.asarray(frozen["pp"], dtype=np.float64),
            "mlp": np.asarray(frozen["mlp"], dtype=np.float64),
            "evidence": "post-test endpoint-matched PP-X development",
        })
    return settings


def matwi_setting() -> dict:
    from matwi_tool_life_untouched import DATA

    frame = pd.read_csv(DATA)
    tools, _ = clean_tools(frame)
    identifiers = sorted(str(int(value)) for value in frame["Set"].unique())
    permutation = np.asarray(identifiers)[
        np.random.default_rng(42).permutation(len(identifiers))
    ]
    development_units = [unit for unit in permutation[:12] if unit in tools]
    test_units = [unit for unit in permutation[12:] if unit in tools]
    max_length = float(max(len(tools[unit]) for unit in development_units))
    frozen = np.load(
        ROOT / "results" / "matwi_tool_life_untouched_v1"
        / "predictions_frozen_before_scoring.npz",
        allow_pickle=True,
    )
    result = json.loads((
        ROOT / "results" / "matwi_tool_life_untouched_v1" / "results.json"
    ).read_text())
    fraction = float(result["pp_selection"]["fraction"])
    development = make_rows(
        tools, development_units, tail_fraction=fraction,
        max_development_length=max_length,
    )
    test = make_rows(
        tools, test_units, tail_fraction=0.30,
        max_development_length=max_length,
    )
    fold_parts = []
    for held_out in folds(development_units):
        fit_units = [unit for unit in development_units if unit not in held_out]
        fold_parts.append((
            make_rows(
                tools, fit_units, tail_fraction=fraction,
                max_development_length=max_length,
            ),
            make_rows(
                tools, held_out, tail_fraction=0.30,
                max_development_length=max_length,
            ),
        ))
    return {
        "cohort": "MATWI",
        "setting": "tool-life",
        "train": fold_parts[0][0],
        "validation": fold_parts[0][1],
        "development": development,
        "test": test,
        "fold_parts": fold_parts,
        "ppx": np.asarray(frozen["pp"], dtype=np.float64),
        "mlp": np.asarray(frozen["mlp"], dtype=np.float64),
        "evidence": "untouched PP core; no approved extra PP-X executor",
    }


def misata_setting() -> dict:
    from misata_machine_untouched import DATA

    frame = pd.read_csv(DATA)
    units, modes, controls, max_length = misata_prepare(frame)
    train_ids = sorted(
        str(value) for value in frame.loc[frame.split == "train", "unit_id"].unique()
    )
    test_ids = sorted(
        str(value) for value in frame.loc[frame.split == "test", "unit_id"].unique()
    )
    order = np.asarray(train_ids)[np.random.default_rng(42).permutation(len(train_ids))]
    fit_ids, validation_ids = list(order[:64]), list(order[64:])
    train = misata_rows(units, fit_ids, modes, controls, max_length, 0.30, stride=3)
    validation = misata_rows(
        units, validation_ids, modes, controls, max_length, 0.30
    )
    development = misata_rows(
        units, train_ids, modes, controls, max_length, 0.30, stride=3
    )
    test = misata_rows(units, test_ids, modes, controls, max_length, 0.30)
    ppx = np.load(
        ROOT / "results" / "misata_ppx_v1" / "predictions.npz",
        allow_pickle=True,
    )
    base = np.load(
        ROOT / "results" / "misata_machine_untouched_v1"
        / "predictions_frozen_before_scoring.npz",
        allow_pickle=True,
    )
    return {
        "cohort": "Misata",
        "setting": "machine-degradation",
        "train": train,
        "validation": validation,
        "development": development,
        "test": test,
        "ppx": np.asarray(ppx["ppx"], dtype=np.float64),
        "mlp": np.asarray(base["mlp"], dtype=np.float64),
        "evidence": "post-outcome PP-X executor development",
    }


def summarize(rows: list[dict], model: str) -> dict:
    r2 = np.asarray([row["metrics"][model]["r2"] for row in rows])
    seed_r2 = np.concatenate([
        np.asarray(row["seed_metrics"][model]["r2"]) for row in rows
    ])
    seed_sd = np.asarray([
        row["seed_metrics"][model]["r2_sd"] for row in rows
    ])
    strict_wins = 0
    ties = 0
    for row in rows:
        values = {name: row["metrics"][name]["r2"] for name in row["metrics"]}
        best = max(values.values())
        if abs(values[model] - best) <= 1e-12:
            if sum(abs(value - best) <= 1e-12 for value in values.values()) == 1:
                strict_wins += 1
            else:
                ties += 1
    return {
        "mean_setting_r2": float(np.mean(r2)),
        "minimum_setting_r2": float(np.min(r2)),
        "positive_r2_settings": int(np.sum(r2 > 0)),
        "strict_wins": strict_wins,
        "best_ties": ties,
        "mean_seed_r2": float(np.mean(seed_r2)),
        "minimum_seed_r2": float(np.min(seed_r2)),
        "mean_within_setting_seed_r2_sd": float(np.mean(seed_sd)),
    }


def render_markdown(payload: dict) -> str:
    labels = {
        "mlp": "MLP",
        "engression": "Engression",
        "ppx": "PP-X",
        "cdcr_ppx": "CDCR-PPX",
    }
    lines = [
        "# 외부 코호트 4모형 동일-split 비교 결과\n\n",
        "> 이 결과는 이미 outcome이 열린 코호트의 retrospective stress test다. "
        "새 외부 확증으로 해석하지 않는다.\n\n",
        "| 코호트 / 설정 | MLP R2 | Engression R2 | PP-X R2 | CDCR-PPX R2 | 최고 |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for row in payload["settings"]:
        scores = {name: row["metrics"][name]["r2"] for name in labels}
        best = max(scores.values())
        winners = ", ".join(labels[name] for name, value in scores.items()
                            if abs(value - best) <= 1e-12)
        lines.append(
            f"| {row['cohort']} / {row['setting']} | {scores['mlp']:.3f} | "
            f"{scores['engression']:.3f} | {scores['ppx']:.3f} | "
            f"{scores['cdcr_ppx']:.3f} | **{winners}** |\n"
        )
    lines.extend([
        "\n## 동일 설정 요약\n\n",
        "| 모형 | 평균 R2 | 최저 R2 | 양수 설정 | 단독 승 | 공동 승 | 평균 seed R2 | 최저 seed R2 | seed SD 평균 |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ])
    for name in labels:
        row = payload["summary"][name]
        lines.append(
            f"| {labels[name]} | {row['mean_setting_r2']:.3f} | "
            f"{row['minimum_setting_r2']:.3f} | {row['positive_r2_settings']}/5 | "
            f"{row['strict_wins']} | {row['best_ties']} | "
            f"{row['mean_seed_r2']:.3f} | {row['minimum_seed_r2']:.3f} | "
            f"{row['mean_within_setting_seed_r2_sd']:.3f} |\n"
        )
    lines.extend([
        "\n## 증거 범위\n\n",
        "- Axial-fan 세 행은 하나의 코호트에서 나온 상관된 configuration이다.\n",
        "- Axial-fan PP-X는 untouched 실패 뒤 개발된 endpoint-matched 복구형이다.\n",
        "- MATWI는 별도 executor가 승인되지 않아 PP core가 적용 가능한 PP-X route다.\n",
        "- Misata PP-X executor와 CDCR 비교는 outcome 개봉 뒤 분석이다.\n",
        "- 따라서 이 표는 모델의 외부 stress-test 결과이며 prospective 우월성 근거가 아니다.\n",
    ])
    return "".join(lines)


def main() -> None:
    torch.set_num_threads(2)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    started = time.time()
    head = fit_development_head()
    settings = axial_settings() + [matwi_setting(), misata_setting()]
    results = []
    arrays = {}
    for item in settings:
        engression_predictions, audit = run_engression(
            item["train"],
            item["validation"],
            item["development"],
            item["test"],
            fold_parts=item.get("fold_parts"),
        )
        ppx = item["ppx"]
        mlp = item["mlp"]
        y = np.asarray(item["test"]["y"], dtype=np.float64)
        if not (ppx.shape == mlp.shape == engression_predictions.shape):
            raise RuntimeError(
                f"prediction shape mismatch for {item['cohort']} {item['setting']}: "
                f"{ppx.shape}, {mlp.shape}, {engression_predictions.shape}"
            )
        cdcr = predict_cross_domain_consensus(head, ppx)
        matrices = {
            "mlp": mlp,
            "engression": engression_predictions,
            "ppx": ppx,
            "cdcr_ppx": cdcr.seeds,
        }
        row = {
            "cohort": item["cohort"],
            "setting": item["setting"],
            "evidence": item["evidence"],
            "n_rows": len(y),
            "engression_audit": audit,
            "cdcr_active_residual": cdcr.active_residual,
            "cdcr_q90_seed_disagreement": cdcr.q90_seed_disagreement,
            "metrics": {
                name: metric(y, matrix.mean(axis=0))
                for name, matrix in matrices.items()
            },
            "seed_metrics": {},
        }
        for name, matrix in matrices.items():
            values = np.asarray([metric(y, prediction)["r2"] for prediction in matrix])
            row["seed_metrics"][name] = {
                "r2": values.tolist(),
                "r2_mean": float(np.mean(values)),
                "r2_min": float(np.min(values)),
                "r2_sd": float(np.std(values, ddof=1)),
            }
        key = f"{item['cohort']}_{item['setting']}".lower().replace("-", "_")
        arrays[f"{key}_truth"] = y
        for name, matrix in matrices.items():
            arrays[f"{key}_{name}"] = matrix
        results.append(row)
        print(
            "SCORED", item["cohort"], item["setting"],
            {name: round(value["r2"], 6) for name, value in row["metrics"].items()},
            flush=True,
        )
    payload = {
        "status": "retrospective matched external-cohort stress test",
        "protocol": PROTOCOL,
        "engression_grid_size": len(GRID),
        "seeds": list(SEEDS),
        "elapsed_seconds": time.time() - started,
        "settings": results,
        "summary": {
            name: summarize(results, name)
            for name in ("mlp", "engression", "ppx", "cdcr_ppx")
        },
        "interpretation_firewall": [
            "All five outcomes were opened before this benchmark.",
            "Axial-fan PP-X is post-test endpoint-matched development.",
            "MATWI uses the applicable PP core because no extra PP-X executor was approved.",
            "Misata PP-X and all CDCR results are retrospective.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    (OUT / "RESULTS_KO.md").write_text(render_markdown(payload))
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
