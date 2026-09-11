#!/usr/bin/env python3
"""Extract Tongji CY45-05 discharge capacities for CCMR v2.0."""
from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tongji_ppx/Tongji.zip"
OUT = ROOT / "data/external/tongji_cy45_05"
PROTOCOL = ROOT / "protocols/TONGJI_CY45_05_CCMR_V20_PROTOCOL.md"


def series_from_record(record):
    cycles, capacities = [], []
    for row in record.get("cycle_data", []):
        values = np.asarray(
            row.get("discharge_capacity_in_Ah") or [], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if values.size == 0 or float(np.max(values)) <= 0:
            continue
        cycles.append(float(row["cycle_number"]))
        capacities.append(float(np.max(values)))
    if not cycles:
        return None
    unique = np.unique(cycles)
    collapsed = [
        float(np.median(np.asarray(capacities)[np.asarray(cycles) == cycle]))
        for cycle in unique
    ]
    return np.asarray(unique, dtype=float), np.asarray(collapsed, dtype=float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "capacity.csv"
    if output.exists():
        raise RuntimeError("refusing to overwrite Tongji CY45-05 extraction")
    rows = []
    members = []
    with zipfile.ZipFile(DATA) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.endswith(".pkl") and "CY45-05" in Path(name).name
        )
        if len(names) != 56:
            raise RuntimeError(f"expected 56 CY45-05 pkls, found {len(names)}")
        for name in names:
            record = pickle.loads(archive.read(name))
            parsed = series_from_record(record)
            if parsed is None:
                continue
            cycles, capacities = parsed
            cell = Path(name).stem
            members.append(name)
            for cycle, capacity in zip(cycles, capacities):
                rows.append({
                    "cell": cell,
                    "member": name,
                    "cycle": float(cycle),
                    "capacity_ah": float(capacity),
                })
    frame = pd.DataFrame(rows).sort_values(["cell", "cycle"])
    frame.to_csv(output, index=False)
    manifest = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "archive": str(DATA.relative_to(ROOT)),
        "archive_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "population": "filename contains CY45-05",
        "n_members": len(members),
        "members": members,
        "cells": int(frame["cell"].nunique()),
        "rows": len(frame),
        "capacity_csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({
        "cells": manifest["cells"],
        "rows": manifest["rows"],
        "members": manifest["n_members"],
        "sha256": manifest["capacity_csv_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
