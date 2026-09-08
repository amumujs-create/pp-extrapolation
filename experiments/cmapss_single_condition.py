"""Portable causal representation for single-condition C-MAPSS subsets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 30
INITIAL_BASELINE = 20
RUL_CAP = 125.0
N_SENSORS = 21
MAX_CLUSTERS = 6


def data_directory() -> Path:
    candidates = []
    if os.environ.get("CMAPSS_DATA_DIR"):
        candidates.append(Path(os.environ["CMAPSS_DATA_DIR"]))
    candidates += [ROOT / "data" / "cmapss", ROOT.parent / "ca-css-ncmapss" / "data" / "cmapss"]
    for path in candidates:
        if (path / "train_FD001.txt").is_file():
            return path
    raise FileNotFoundError(
        "C-MAPSS files not found; set CMAPSS_DATA_DIR or place train/test/RUL files in data/cmapss"
    )


def load_fd(fd: str):
    path = data_directory()
    train = np.loadtxt(path / f"train_{fd}.txt", dtype=np.float64)
    test = np.loadtxt(path / f"test_{fd}.txt", dtype=np.float64)
    truth = np.loadtxt(path / f"RUL_{fd}.txt", dtype=np.float64).reshape(-1)
    for value in (train, test):
        if value.ndim != 2 or value.shape[1] != 26 or not np.isfinite(value).all():
            raise ValueError(f"invalid {fd} C-MAPSS table")
    if len(np.unique(test[:, 0].astype(int))) != len(truth):
        raise ValueError("official RUL count does not match test engine count")
    return train, test, truth


def split_table_units(table, *, seed=42, validation_fraction=0.2):
    units = np.unique(table[:, 0].astype(np.int32))
    rng = np.random.default_rng(seed)
    n_validation = max(1, int(round(len(units) * validation_fraction)))
    validation_units = np.sort(rng.choice(units, n_validation, replace=False))
    mask = np.isin(table[:, 0].astype(np.int32), validation_units)
    return table[~mask], table[mask], units[~np.isin(units, validation_units)], validation_units


@dataclass(frozen=True)
class SingleConditionNormalizer:
    op_center: np.ndarray
    op_scale: np.ndarray
    sensor_center: np.ndarray
    sensor_scale: np.ndarray

    @classmethod
    def fit(cls, table):
        op = table[:, 2:5]
        sensors = table[:, 5:26]
        op_scale = op.std(0)
        op_scale[op_scale < 1e-8] = 1.0
        sensor_scale = sensors.std(0)
        sensor_scale[sensor_scale < 1e-8] = 1.0
        return cls(op.mean(0), op_scale, sensors.mean(0), sensor_scale)

    def transform(self, table):
        z_sensor = (table[:, 5:26] - self.sensor_center) / self.sensor_scale
        z_op = (table[:, 2:5] - self.op_center) / self.op_scale
        occupancy = np.zeros((len(table), MAX_CLUSTERS), dtype=np.float64)
        occupancy[:, 0] = 1.0
        return np.column_stack((z_sensor, z_op, occupancy))


def _feature_at(group, normalized, endpoint):
    start = max(0, endpoint - WINDOW + 1)
    block = normalized[start:endpoint + 1]
    if len(block) < WINDOW:
        block = np.vstack((np.repeat(block[:1], WINDOW - len(block), axis=0), block))
    sensors = block[:, :N_SENSORS]
    position = np.arange(WINDOW, dtype=np.float64)
    position -= position.mean()
    slope = position @ sensors / np.sum(position * position)
    initial = normalized[:min(INITIAL_BASELINE, endpoint + 1), :N_SENSORS].mean(0)
    return np.concatenate((
        sensors[-1], sensors.mean(0), sensors.std(0), slope, sensors[-1] - initial,
        block[-1, N_SENSORS:N_SENSORS + 3], block[:, N_SENSORS + 3:].mean(0),
        [np.log1p(group[endpoint, 1])],
    )).astype(np.float32)


def make_rows(table, normalizer, *, stride=5, final_only=False):
    normalized = normalizer.transform(table)
    features, targets, units = [], [], []
    for unit in sorted(np.unique(table[:, 0].astype(np.int32))):
        index = np.flatnonzero(table[:, 0].astype(np.int32) == unit)
        index = index[np.argsort(table[index, 1])]
        group, z_group = table[index], normalized[index]
        if final_only:
            endpoints = [len(group) - 1]
        else:
            first = min(WINDOW - 1, len(group) - 1)
            endpoints = list(range(first, len(group), int(stride)))
            if endpoints[-1] != len(group) - 1:
                endpoints.append(len(group) - 1)
        for endpoint in endpoints:
            features.append(_feature_at(group, z_group, endpoint))
            targets.append(min(RUL_CAP, group[-1, 1] - group[endpoint, 1]))
            units.append(unit)
    return {"x": np.asarray(features), "y": np.asarray(targets), "units": np.asarray(units)}
