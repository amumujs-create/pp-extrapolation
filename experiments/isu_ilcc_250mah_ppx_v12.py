#!/usr/bin/env python3
"""Frozen PP-X v1.2 evaluation on the ISU–ILCC 250 mAh cohort."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import uconn_ilcc_ppx_v12 as core

BASE = ROOT / "data/isu_ilcc_250mah_ppx_v12"
CAPACITY_ROOT = BASE / "capacity_fade"
VALID = BASE / "Valid_cells.csv"


def numeric_id(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"G(\d+)C(\d+)", value.strip())
    if match is None:
        raise ValueError(f"invalid cell ID: {value!r}")
    return int(match.group(1)), int(match.group(2))


def valid_ids() -> list[str]:
    with VALID.open(newline="") as handle:
        rows = list(csv.reader(handle))
    values = []
    for row in rows:
        for value in row:
            if re.fullmatch(r"G\d+C\d+", value.strip()):
                values.append(value.strip())
    return sorted(set(values), key=numeric_id)


def load_cells():
    paths = {}
    for path in CAPACITY_ROOT.glob("**/*.csv"):
        if re.fullmatch(r"G\d+C\d+", path.stem):
            # Release 2.0 is a later payload for cells absent from release 1.0.
            if path.stem in paths:
                raise RuntimeError(f"duplicate cell trajectory: {path.stem}")
            paths[path.stem] = path
    cells = {}
    missing = []
    for cell in valid_ids():
        path = paths.get(cell)
        if path is None:
            missing.append(cell)
            continue
        time, capacity = [], []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    t = float(row["Time"])
                    q = float(row["Capacity"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(t) and np.isfinite(q):
                    time.append(t)
                    capacity.append(q)
        order = np.argsort(time, kind="stable")
        ordered_time = np.asarray(time, dtype=float)[order]
        ordered_capacity = np.asarray(capacity, dtype=float)[order]
        unique_time = np.unique(ordered_time)
        cells[cell] = {
            "cycle": unique_time,
            "capacity": np.asarray([
                np.median(ordered_capacity[ordered_time == value])
                for value in unique_time
            ], dtype=float),
        }
    if missing:
        raise RuntimeError(f"valid-cell files missing: {missing}")
    return cells


def eligibility(cell, series):
    capacity = series["capacity"]
    hits = np.flatnonzero(capacity <= 0.200)
    crossing = int(hits[0]) if len(hits) else None
    return {
        "cell": cell,
        "observations": int(len(capacity)),
        "threshold_ah": 0.200,
        "crossing_index": crossing,
        "eligible": bool(
            len(capacity) >= 10 and crossing is not None and crossing >= core.HISTORY
        ),
    }


def split_ids(eligible):
    eligible = sorted(eligible, key=numeric_id)
    n = len(eligible)
    a, b = int(0.6 * n), int(0.8 * n)
    return {
        "train": eligible[:a],
        "validation": eligible[a:b],
        "test": eligible[b:],
    }


if __name__ == "__main__":
    core.DATA = BASE / "capacity_fade.zip"
    core.OUT = ROOT / "results/isu_ilcc_250mah_ppx_v12"
    core.MIN_ELIGIBLE = 100
    core.MIN_OUTER_ROWS = 100
    core.DATASET_NAME = "ISU–ILCC 250 mAh NMC/graphite pouch cells"
    core.PROTOCOL_NAME = "ISU_ILCC_250MAH_PPX_V12_PROTOCOL"
    core.load_cells = load_cells
    core.eligibility = eligibility
    core.split_ids = split_ids
    core.main()
