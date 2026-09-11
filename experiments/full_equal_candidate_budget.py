#!/usr/bin/env python3
"""Restartable 9-setting, 30-validation-candidate baseline benchmark.

PP-X is deliberately a frozen, heterogeneous reference and is never counted as
an equal-budget candidate executor.  Expensive/optional baselines are imported
lazily so that protocol audits and artifact imports need no extra dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"),
                str(ROOT / ".benchmark_deps"), str(ROOT.parent / "ca-css-ncmapss")]

import extrapolation_competitors_matr as neural
from extrapolation_competitors_all import datasets as generic_datasets
from pp_extrapolation import regression_metrics, select_affine_initialization
from pp_extrapolation.model import transform_features

OUT = ROOT / "results/full_equal_candidate_budget_v1"
SEARCH_SEED = 42
REFIT_SEEDS = (42, 43, 44, 45, 46)
MODELS = ("plain_mlp", "ft_transformer", "vrex", "groupdro", "monotone",
          "engression", "linear_tail_rbf", "svgp")
SETTINGS = ("hust", "virkler", "nasa", "sunwoda", "rwth", "matr",
            "matr_batch2", "ncmapss", "mich")
PPX_ARTIFACTS = {
    "hust": "results/hust_regime_transport_pp_v1/results.json",
    "virkler": "results/support_gated_cross_domain_v1/results.json",
    "nasa": "results/nasa_causal_multiscale_pp_v1/results.json",
    "sunwoda": "results/bq_pp_matched_controls_v1/results.json",
    "rwth": "results/bq_pp_matched_controls_v1/results.json",
    "matr": "results/matr_pp_validation_calibration_v1/results.json",
    "matr_batch2": "results/matr_batch2_pp_five_seed_replay/results.json",
    "ncmapss": "results/ncmapss_pp_multiscale_v1/results.json",
    "mich": "results/bq_dual_scale_final_replay_v1/results.json",
}

# Every list contains 30 distinct configurations.  These are protocol objects;
# executors may translate keys to package-specific arguments.
NN30 = [{"width": w, "depth": d, "lr": lr, "weight_decay": wd, "penalty": p}
        for w, d in ((16, 1), (32, 1), (32, 2), (64, 2), (64, 3))
        for lr, wd, p in ((2e-4, .1, .03), (5e-4, .1, .03),
                          (1e-3, .1, .03), (5e-4, 2., .03),
                          (5e-4, .1, .3), (5e-4, .1, 3.))]
FT30 = [{"d_token": d, "blocks": b, "heads": h, "lr": lr, "weight_decay": wd}
        for d, b, h in ((16, 1, 2), (32, 2, 4), (64, 2, 4))
        for lr, wd in ((2e-4, .01), (5e-4, .01), (1e-3, .01),
                       (2e-4, .1), (5e-4, .1), (1e-3, .1),
                       (2e-4, 1.), (5e-4, 1.), (1e-3, 1.), (2e-3, .1))]
ENG30 = [{"hidden_dim": h, "lr": lr, "beta": beta, "layers": layers, "epochs": ep}
         for h in (32, 64, 128) for lr, beta, layers, ep in
         ((1e-3, .5, 2, 250), (5e-3, .5, 2, 250),
          (1e-3, 1., 2, 500), (5e-3, 1., 3, 500),
          (2e-3, .1, 3, 500), (2e-3, 2., 3, 750),
          (5e-4, .5, 4, 500), (1e-2, .5, 2, 250),
          (5e-4, 1., 2, 750), (1e-3, 2., 4, 750))]
RBF30 = [{"gamma": g, "alpha": a, "components": c}
         for g in (.01, .03, .1, .3, 1.) for a, c in
         ((.03, 256), (.1, 384), (1., 384), (10., 384), (100., 512), (3., 768))]
SVGP30 = [{"lr": lr, "lengthscale": length, "noise": noise}
          for lr in (.003, .01, .03) for length in (.3, 1., 3., 10., 30.)
          for noise in (.01, .1)]
CANDIDATES = {m: (FT30 if m == "ft_transformer" else ENG30 if m == "engression"
                  else RBF30 if m == "linear_tail_rbf" else SVGP30 if m == "svgp"
                  else NN30) for m in MODELS}


def canonical_hash(value: Any) -> str:
    """Stable hash for arrays including row order, shape and dtype."""
    h = hashlib.sha256()
    a = np.asarray(value)
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    h.update(a.tobytes() if a.dtype.kind != "O" else
             json.dumps(a.tolist(), ensure_ascii=False, separators=(",", ":")).encode())
    return h.hexdigest()


def split_hashes(parts: tuple[dict, dict, dict]) -> dict[str, dict[str, str]]:
    return {name: {k: canonical_hash(part[k]) for k in ("x", "y", "groups")}
            for name, part in zip(("train", "validation", "test"), parts)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_settings(wanted: tuple[str, ...]) -> dict[str, Any]:
    available = generic_datasets()
    if "nasa" in wanted:
        from run_affine_tail_external_nasa_health_v2 import prepare_folds
        folds, _ = prepare_folds()
        available["nasa"] = [(f["train"], f["validation"], f["test"]) for f in folds]
    if "ncmapss" in wanted:
        from apps.ncmapss_data_utils import FEATURE_COLS
        from ncmapss_pp_benchmark import rows
        from ncmapss_tra_quantile_split import make_tra_hard_split
        split = make_tra_hard_split(ROOT / "data/N-CMAPSS_DS02-006.h5",
                                    max_windows_per_unit=1500, random_seed=SEARCH_SEED)
        names = list(FEATURE_COLS)
        available["ncmapss"] = (rows(split.train, names), rows(split.val, names),
                                rows(split.test, names))
    return {name: available[name] for name in wanted}


def _plain_fit(parts, cfg, seed, return_test=True):
    """Plain equal-group-weight MLP; intentionally shares the established adapter."""
    affine = select_affine_initialization(parts[0], parts[1])
    packed = (cfg["width"], cfg["depth"], cfg["lr"], cfg["weight_decay"], 0.0)
    # The established monotone branch with zero penalty is exactly its plain MLP.
    return neural.fit_neural("monotone", packed, seed, parts, affine, return_test)


def _neural_fit(model, parts, cfg, seed, return_test=True):
    affine = select_affine_initialization(parts[0], parts[1])
    packed = (cfg["width"], cfg["depth"], cfg["lr"], cfg["weight_decay"], cfg["penalty"])
    return neural.fit_neural(model, packed, seed, parts, affine, return_test)


def _rbf_fit(parts, cfg, seed, return_test=True):
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    affine = select_affine_initialization(parts[0], parts[1])
    z = [transform_features(p["x"], affine["center"], affine["scale"]) for p in parts]
    rff = RBFSampler(gamma=cfg["gamma"], n_components=cfg["components"],
                     random_state=seed).fit(z[0])
    design = lambda x: np.column_stack((np.ones(len(x)), x, rff.transform(x)))
    model = Ridge(alpha=cfg["alpha"], fit_intercept=False).fit(design(z[0]), parts[0]["y"])
    cap = float(affine["target_scale"])
    vp = np.clip(model.predict(design(z[1])), 0, cap)
    info = {"validation_mse": float(np.mean((vp - parts[1]["y"]) ** 2)),
            "parameters": int(sum(np.size(v) for v in (model.coef_, model.intercept_)))}
    return info, np.clip(model.predict(design(z[2])), 0, cap) if return_test else None


def _ft_fit(parts, cfg, seed, return_test=True):
    from extended_nn_benchmark import train
    affine = select_affine_initialization(parts[0], parts[1])
    translated = {"width": cfg["d_token"], "depth": cfg["blocks"],
                  "lr": cfg["lr"], "wd": cfg["weight_decay"]}
    model, test_x, info = train("ft_transformer", translated, seed, parts, affine)
    if not return_test:
        return info, None
    model.eval()
    with torch.no_grad():
        raw = torch.cat([model(batch) for batch in test_x.split(512)]).numpy()
    return info, np.clip(raw * float(affine["target_scale"]),
                         0, float(affine["target_scale"]))


def executor(model: str) -> Callable:
    if model == "plain_mlp":
        return _plain_fit
    if model in ("vrex", "groupdro", "monotone"):
        return lambda parts, cfg, seed, return_test=True: _neural_fit(
            model, parts, cfg, seed, return_test)
    if model == "linear_tail_rbf":
        return _rbf_fit
    if model == "engression":
        from engression_all_positive import fit_predict
        return lambda parts, cfg, seed, return_test=True: (
            {"validation_mse": (r := fit_predict(*parts, tuple(cfg.values()), seed))[0]},
            r[1] if return_test else None)
    if model == "svgp":
        from mich_svgp_extension import fit_predict
        return lambda parts, cfg, seed, return_test=True: (
            {"validation_mse": (r := fit_predict(parts, tuple(cfg.values()), seed))[0],
             "epochs": r[2], "inducing_points": r[3]}, r[1] if return_test else None)
    if model == "ft_transformer":
        return _ft_fit
    raise KeyError(model)


def import_completed(setting: str, model: str, parts, source: Path, dest: Path) -> dict:
    """Verify and import a completed V-REx/GroupDRO/monotone 30c artifact."""
    payload = json.loads(source.read_text())
    row = payload.get("datasets", {}).get(setting, {}).get(model)
    if row is None:
        raise KeyError(f"{setting}/{model} absent from {source}")
    search = row.get("search", [])
    if len(search) != 30 or row.get("search_budget", 30) != 30:
        raise ValueError("source is not an exact 30-candidate artifact")
    npz_options = [source.parent / "predictions" / f"{setting}_{model}.npz",
                   source.parent / f"{setting}_{model}_predictions.npz"]
    npz = next((p for p in npz_options if p.exists()), None)
    if npz is None:
        raise FileNotFoundError(f"prediction artifact missing: {npz_options}")
    z = np.load(npz, allow_pickle=False)
    expected = parts[2]
    if not np.array_equal(z["y"], expected["y"]) or not np.array_equal(z["groups"], expected["groups"]):
        raise ValueError("prediction rows are not aligned to the current test split")
    pred = np.asarray(z["prediction"])
    if pred.shape != (5, len(expected["y"])):
        raise ValueError(f"expected row-aligned (5, {len(expected['y'])}) predictions, got {pred.shape}")
    target = dest / "predictions.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, prediction=pred, y=z["y"], groups=z["groups"])
    return {"status": "imported_verified", "source": str(source.relative_to(ROOT)),
            "candidate_count": 30, "search_seed": SEARCH_SEED,
            "refit_seeds": list(REFIT_SEEDS), "split_hashes": split_hashes(parts),
            "prediction_sha256": canonical_hash(pred), "result": row}


def run_one(setting: str, model: str, parts, out: Path, limit_candidates: int | None = None) -> dict:
    candidates = CANDIDATES[model]
    if len(candidates) != 30 or len({json.dumps(x, sort_keys=True) for x in candidates}) != 30:
        raise AssertionError(f"{model}: candidate contract is not 30 distinct configurations")
    used = candidates if limit_candidates is None else candidates[:limit_candidates]
    fit = executor(model); search = []; started = time.monotonic()
    for index, cfg in enumerate(used):
        t = time.monotonic(); info, _ = fit(parts, cfg, SEARCH_SEED, False)
        search.append({"candidate": index, "config": cfg, "seconds": time.monotonic() - t, **info})
    chosen = min(search, key=lambda x: x["validation_mse"])["config"]
    predictions, runs = [], []
    for seed in REFIT_SEEDS:
        t = time.monotonic(); info, pred = fit(parts, chosen, seed, True)
        pred = np.asarray(pred)
        if pred.shape != np.asarray(parts[2]["y"]).shape:
            raise ValueError("executor returned non-row-aligned predictions")
        predictions.append(pred)
        runs.append({"seed": seed, "seconds": time.monotonic() - t, **info,
                     "metrics": regression_metrics(parts[2]["y"], pred, parts[2]["groups"])})
    matrix = np.asarray(predictions)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", prediction=matrix,
                        y=parts[2]["y"], groups=parts[2]["groups"])
    return {"status": "complete" if limit_candidates is None else "smoke",
            "candidate_count": len(used), "contract_candidate_count": 30,
            "search_seed": SEARCH_SEED, "refit_seeds": list(REFIT_SEEDS),
            "selected_config": chosen, "search": search, "runs": runs,
            "ensemble": regression_metrics(parts[2]["y"], matrix.mean(0), parts[2]["groups"]),
            "split_hashes": split_hashes(parts), "prediction_sha256": canonical_hash(matrix),
            "seconds": time.monotonic() - started}


def synthetic_parts(seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for lo, hi, n in ((0, .6, 48), (.6, .8, 16), (.8, 1., 16)):
        x = rng.uniform(lo, hi, (n, 3)).astype(np.float32)
        parts.append({"x": x, "y": (2 * x[:, 0] + .2 * x[:, 1]).astype(np.float32),
                      "groups": np.repeat(np.arange(4), n // 4)})
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--import-30c", type=Path, help="existing final_30... results.json")
    parser.add_argument("--smoke", action="store_true", help="synthetic plain-MLP, two candidates")
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings = tuple(x for x in args.settings.split(",") if x)
    models = tuple(x for x in args.models.split(",") if x)
    if args.smoke:
        settings, models, loaded = ("smoke",), ("plain_mlp",), {"smoke": synthetic_parts()}
    else:
        loaded = load_settings(settings)
    result_path = args.out / "results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "protocol_version": 1, "candidate_budget": 30, "search_seed": SEARCH_SEED,
        "selected_refit_seeds": list(REFIT_SEEDS),
        "ppx": {"equal_candidate_budget": False, "executor": "heterogeneous_frozen_reference",
                "artifacts": PPX_ARTIFACTS}, "settings": {}}
    for setting in settings:
        units = loaded[setting] if setting == "nasa" else [loaded[setting]]
        result["settings"].setdefault(setting, {})
        for fold, parts in enumerate(units):
            unit = f"fold_{fold}" if setting == "nasa" else "main"
            result["settings"][setting].setdefault(unit, {})
            for model in models:
                if model in result["settings"][setting][unit]:
                    continue
                destination = args.out / setting / unit / model
                if args.import_30c and model in ("vrex", "groupdro", "monotone"):
                    if setting == "nasa":
                        raise ValueError("NASA imports must provide fold-level artifacts")
                    row = import_completed(setting, model, parts, args.import_30c, destination)
                else:
                    row = run_one(setting, model, parts, destination, 2 if args.smoke else None)
                result["settings"][setting][unit][model] = row
                atomic_json(result_path, result)
                print("DONE", setting, unit, model, flush=True)


if __name__ == "__main__":
    main()
