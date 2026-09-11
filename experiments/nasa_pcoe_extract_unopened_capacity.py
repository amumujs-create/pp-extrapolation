#!/usr/bin/env python3
"""Extract selected unopened NASA PCoE discharge capacities to compact CSV."""
from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/nasa_pcoe_new_unopened/battery_data_set.zip"
OUT = ROOT / "data/nasa_pcoe_new_unopened/unopened_capacity_summary.csv"
SELECTED = {
    "2. BatteryAgingARC_25_26_27_28_P1.zip": {
        "B0025", "B0026", "B0027", "B0028"
    },
    "3. BatteryAgingARC_25-44.zip": {
        "B0033", "B0034", "B0036", "B0038", "B0039", "B0040",
        "B0041", "B0042", "B0043", "B0044",
    },
    "5. BatteryAgingARC_49_50_51_52.zip": {
        "B0049", "B0050", "B0051", "B0052"
    },
}


def capacities(payload):
    mat = loadmat(BytesIO(payload), squeeze_me=True, struct_as_record=False)
    key = next(name for name in mat if not name.startswith("__"))
    cycles = np.atleast_1d(mat[key].cycle)
    values = []
    for cycle in cycles:
        if str(getattr(cycle, "type", "")).lower() != "discharge":
            continue
        data = cycle.data
        if not hasattr(data, "Capacity"):
            continue
        capacity = np.asarray(data.Capacity, dtype=np.float64).reshape(-1)
        if capacity.size and np.isfinite(capacity[0]) and capacity[0] > 0:
            values.append(float(capacity[0]))
    return values


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    rows = []
    with ZipFile(ARCHIVE) as outer:
        for outer_name in outer.namelist():
            label = next(
                (name for name in SELECTED if outer_name.endswith(name)), None
            )
            if label is None:
                continue
            with ZipFile(BytesIO(outer.read(outer_name))) as inner:
                for member in inner.namelist():
                    cell = Path(member).stem
                    if cell not in SELECTED[label] or not member.endswith(".mat"):
                        continue
                    values = capacities(inner.read(member))
                    rows.extend({
                        "cell": cell,
                        "cycle": index,
                        "capacity_ah": value,
                        "source_archive": label,
                    } for index, value in enumerate(values))
                    print(cell, len(values), flush=True)
    expected = set().union(*SELECTED.values())
    observed = {row["cell"] for row in rows}
    if observed != expected:
        raise RuntimeError(f"selected cells missing: {sorted(expected - observed)}")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "cell", "cycle", "capacity_ah", "source_archive"
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows, {OUT.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
