#!/usr/bin/env python3
"""Post-hoc journal evidence audit for PP support geometry.

This script does not fit or select a predictor.  It reuses the frozen matched
PP/MLP predictions and asks whether several train-only notions of support
explain physical-unit error differences.  Test outcomes are used only for the
reported retrospective association analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from cross_domain_mechanism_study import all_datasets
from pp_extrapolation import audit_convex_hull_support

PREDICTIONS = ROOT / "results" / "cross_domain_mechanism_v1"
OUTPUT = ROOT / "results" / "journal_support_geometry_v1"
SEED = 20260908


def standardize(train: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-10] = 1.0
    return (train - mean) / scale, (source - mean) / scale


def pca_hull_distance(train: np.ndarray, source: np.ndarray, dimension: int) -> dict:
    train_z, source_z = standardize(np.asarray(train, float), np.asarray(source, float))
    usable = min(dimension, train_z.shape[1], max(1, np.linalg.matrix_rank(train_z)))
    pca = PCA(n_components=usable, svd_solver="full").fit(train_z)
    audit = audit_convex_hull_support(pca.transform(train_z), pca.transform(source_z))
    return {
        "requested_dimension": dimension,
        "used_dimension": usable,
        "explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
        "distance": audit.distance,
        "summary": audit.summary(),
    }


def local_density_distance(train: np.ndarray, source: np.ndarray, k: int = 10) -> dict:
    train_z, source_z = standardize(np.asarray(train, float), np.asarray(source, float))
    rng = np.random.default_rng(SEED)
    if len(train_z) > 5000:
        reference = train_z[rng.choice(len(train_z), 5000, replace=False)]
    else:
        reference = train_z
    use_k = min(k, len(reference) - 1)
    model = NearestNeighbors(n_neighbors=use_k + 1).fit(reference)
    train_radius = model.kneighbors(reference, return_distance=True)[0][:, -1]
    source_radius = model.kneighbors(source_z, n_neighbors=use_k, return_distance=True)[0][:, -1]
    baseline = max(float(np.median(train_radius)), 1e-10)
    ratio = source_radius / baseline
    return {
        "k": use_k,
        "reference_train_rows": len(reference),
        "train_median_k_radius": baseline,
        "ratio": ratio,
        "summary": {
            "median": float(np.median(ratio)),
            "p95": float(np.quantile(ratio, 0.95)),
            "fraction_above_train_median": float(np.mean(ratio > 1.0)),
        },
    }


def support_arrays(train: dict, test: dict) -> tuple[dict, dict]:
    x_train = np.asarray(train["x"], float)
    x_test = np.asarray(test["x"], float)
    one = audit_convex_hull_support(x_train[:, [0]], x_test[:, [0]])
    two = pca_hull_distance(x_train, x_test, 2)
    three = pca_hull_distance(x_train, x_test, 3)
    knn = local_density_distance(x_train, x_test)
    arrays = {
        "ordered_1d_hull": one.distance,
        "pca_2d_hull": two["distance"],
        "pca_3d_hull": three["distance"],
        "knn10_density_ratio": knn["ratio"],
    }
    summary = {
        "n_features": int(x_train.shape[1]),
        "ordered_1d_hull": one.summary(),
        "pca_2d_hull": {**two["summary"], "explained_variance_fraction": two["explained_variance_fraction"]},
        "pca_3d_hull": {**three["summary"], "explained_variance_fraction": three["explained_variance_fraction"]},
        "knn10_density_ratio": knn["summary"],
    }
    return arrays, summary


def paired_units(y: np.ndarray, groups: np.ndarray, plain: np.ndarray, pp: np.ndarray, support: dict) -> list[dict]:
    rows = []
    for unit in np.unique(groups):
        mask = groups == unit
        plain_rmse = float(np.sqrt(np.mean((y[mask] - plain[mask]) ** 2)))
        pp_rmse = float(np.sqrt(np.mean((y[mask] - pp[mask]) ** 2)))
        row = {
            "unit": str(unit),
            "n": int(mask.sum()),
            "plain_rmse": plain_rmse,
            "pp_rmse": pp_rmse,
            "relative_rmse_gain": float((plain_rmse - pp_rmse) / max(plain_rmse, 1e-10)),
        }
        row.update({f"median_{name}": float(np.median(value[mask])) for name, value in support.items()})
        rows.append(row)
    return rows


def correlations(rows: list[dict]) -> dict:
    out = {}
    gain = np.asarray([row["relative_rmse_gain"] for row in rows])
    for key in (
        "median_ordered_1d_hull",
        "median_pca_2d_hull",
        "median_pca_3d_hull",
        "median_knn10_density_ratio",
    ):
        value = np.asarray([row[key] for row in rows])
        if len(value) < 3 or np.allclose(value, value[0]):
            out[key] = {"rho": None, "p": None}
        else:
            stat = spearmanr(value, gain)
            out[key] = {"rho": float(stat.statistic), "p": float(stat.pvalue)}
    return out


def one_dataset(name: str, train: dict, test: dict, prediction_path: Path) -> tuple[dict, list[dict]]:
    saved = np.load(prediction_path, allow_pickle=True)
    y = np.asarray(saved["y"], float)
    groups = np.asarray(saved["groups"])
    if len(y) != len(test["y"]) or not np.array_equal(groups.astype(str), np.asarray(test["groups"]).astype(str)):
        raise RuntimeError(f"{name}: frozen prediction rows do not match reconstructed test rows")
    support, summary = support_arrays(train, test)
    unit_rows = paired_units(y, groups, saved["plain"].mean(0), saved["pp"].mean(0), support)
    result = {
        "n_test_rows": len(y),
        "n_physical_units": len(unit_rows),
        "support": summary,
        "unit_correlations": correlations(unit_rows),
        "mean_relative_rmse_gain": float(np.mean([row["relative_rmse_gain"] for row in unit_rows])),
        "pp_unit_wins": int(sum(row["relative_rmse_gain"] > 0 for row in unit_rows)),
        "units": unit_rows,
    }
    flattened = [{"dataset": name, **row} for row in unit_rows]
    return result, flattened


def nasa_dataset(item: dict, prediction_path: Path) -> tuple[dict, list[dict]]:
    saved = np.load(prediction_path, allow_pickle=True)
    supports = {key: [] for key in ("ordered_1d_hull", "pca_2d_hull", "pca_3d_hull", "knn10_density_ratio")}
    fold_summaries = []
    offset = 0
    for fold in item["folds"]:
        n = len(fold["test"]["y"])
        expected = np.asarray(fold["test"]["groups"]).astype(str)
        actual = np.asarray(saved["groups"])[offset : offset + n].astype(str)
        if not np.array_equal(expected, actual):
            raise RuntimeError("nasa_battery: fold rows do not match frozen predictions")
        arrays, summary = support_arrays(fold["train"], fold["test"])
        for key in supports:
            supports[key].append(arrays[key])
        fold_summaries.append({"test_cell": fold["test_cell"], "support": summary})
        offset += n
    support = {key: np.concatenate(value) for key, value in supports.items()}
    y = np.asarray(saved["y"], float)
    groups = np.asarray(saved["groups"])
    unit_rows = paired_units(y, groups, saved["plain"].mean(0), saved["pp"].mean(0), support)
    result = {
        "n_test_rows": len(y),
        "n_physical_units": len(unit_rows),
        "fold_support": fold_summaries,
        "unit_correlations": correlations(unit_rows),
        "mean_relative_rmse_gain": float(np.mean([row["relative_rmse_gain"] for row in unit_rows])),
        "pp_unit_wins": int(sum(row["relative_rmse_gain"] > 0 for row in unit_rows)),
        "units": unit_rows,
    }
    return result, [{"dataset": "nasa_battery", **row} for row in unit_rows]


def permutation_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = 20000) -> dict:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": None, "permutation_p": None}
    observed = float(spearmanr(x, y).statistic)
    null = np.empty(n)
    for i in range(n):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n + 1))
    return {"rho": observed, "permutation_p": p, "permutations": n}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = all_datasets()
    results = {}
    all_units = []
    for name, item in datasets.items():
        path = PREDICTIONS / f"{name}_predictions.npz"
        if not path.exists():
            continue
        if isinstance(item, dict) and "folds" in item:
            result, rows = nasa_dataset(item, path)
        else:
            result, rows = one_dataset(name, item[0], item[2], path)
        results[name] = result
        all_units.extend(rows)
        print(name, result["pp_unit_wins"], "/", result["n_physical_units"], flush=True)

    # Equal weight per dataset: each domain contributes one median support and
    # one mean physical-unit effect. This avoids large row counts dominating.
    dataset_rows = []
    for name, result in results.items():
        units = result["units"]
        dataset_rows.append({
            "dataset": name,
            "gain": float(np.mean([u["relative_rmse_gain"] for u in units])),
            **{
                key: float(np.median([u[f"median_{key}"] for u in units]))
                for key in ("ordered_1d_hull", "pca_2d_hull", "pca_3d_hull", "knn10_density_ratio")
            },
        })
    rng = np.random.default_rng(SEED)
    gain = np.asarray([row["gain"] for row in dataset_rows])
    cross = {}
    for key in ("ordered_1d_hull", "pca_2d_hull", "pca_3d_hull", "knn10_density_ratio"):
        cross[key] = permutation_spearman(
            np.asarray([row[key] for row in dataset_rows]), gain, rng
        )

    payload = {
        "experiment": "journal_support_geometry_v1",
        "status": "retrospective evidence audit; no model fitting or selection",
        "estimand": "physical-unit relative RMSE gain of frozen PP over matched plain MLP",
        "support_definitions": {
            "ordered_1d_hull": "declared first degradation coordinate, standardized on train",
            "pca_2d_hull": "convex hull after train-only standardized PCA to 2 dimensions",
            "pca_3d_hull": "convex hull after train-only standardized PCA to 3 dimensions",
            "knn10_density_ratio": "test 10-NN radius divided by median train 10-NN radius",
        },
        "datasets": results,
        "equal_dataset_meta_association": cross,
        "dataset_rows": dataset_rows,
        "limitations": [
            "Post-hoc association cannot validate an outcome-free deployment gate.",
            "PCA hulls test geometric sensitivity but their axes are not domain-physical coordinates.",
            "Rows within a physical unit are dependent; inference is summarized at unit and dataset levels.",
            "Convex-hull membership is a support diagnostic, not proof that prediction is reliable.",
        ],
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        OUTPUT / "unit_table.npz",
        dataset=np.asarray([row["dataset"] for row in all_units]),
        unit=np.asarray([row["unit"] for row in all_units]),
        relative_rmse_gain=np.asarray([row["relative_rmse_gain"] for row in all_units]),
        ordered_1d_hull=np.asarray([row["median_ordered_1d_hull"] for row in all_units]),
        pca_2d_hull=np.asarray([row["median_pca_2d_hull"] for row in all_units]),
        pca_3d_hull=np.asarray([row["median_pca_3d_hull"] for row in all_units]),
        knn10_density_ratio=np.asarray([row["median_knn10_density_ratio"] for row in all_units]),
    )
    print(json.dumps(cross, indent=2))


if __name__ == "__main__":
    main()
