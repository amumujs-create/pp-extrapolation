#!/usr/bin/env python3
"""Extract compact reference-discharge capacities from NASA ALT telemetry."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/nasa_alt_external/battery_alt_dataset.zip"
OUT = ROOT / "data/nasa_alt_external/reference_capacity_summary.csv"
USECOLS = ("start_time", "time", "mode", "current_load", "mission_type")


def extract_member(archive: ZipFile, member: str):
    capacity = defaultdict(float)
    count = defaultdict(int)
    first_time = {}
    carry = None
    with archive.open(member) as stream:
        for chunk in pd.read_csv(
            stream, usecols=USECOLS, chunksize=500_000,
            dtype={
                "start_time": "string", "time": "float64", "mode": "float32",
                "current_load": "float32", "mission_type": "float32",
            },
        ):
            take = (
                (chunk["mode"] < -0.5)
                & (chunk["mission_type"] == 0)
                & np.isfinite(chunk["time"])
                & np.isfinite(chunk["current_load"])
                & (chunk["current_load"] > 0)
            )
            frame = chunk.loc[take, ["start_time", "time", "current_load"]]
            if frame.empty:
                continue
            for key, value in frame.groupby("start_time", sort=False).size().items():
                count[str(key)] += int(value)
            for key, value in frame.groupby("start_time", sort=False)["time"].min().items():
                key = str(key)
                first_time[key] = min(first_time.get(key, np.inf), float(value))
            if carry is not None:
                frame = pd.concat([carry, frame], ignore_index=True)
            same = frame["start_time"].eq(frame["start_time"].shift())
            dt = frame["time"].diff()
            valid = same & (dt > 0) & (dt <= 5)
            contribution = (
                0.5
                * (frame["current_load"] + frame["current_load"].shift())
                * dt
                / 3600.0
            )
            summed = contribution[valid].groupby(
                frame.loc[valid, "start_time"], sort=False
            ).sum()
            for key, value in summed.items():
                capacity[str(key)] += float(value)
            carry = frame.iloc[[-1]].copy()
    rows = []
    for start_time in sorted(capacity, key=lambda key: first_time[key]):
        if count[start_time] >= 600 and capacity[start_time] > 0:
            rows.append({
                "start_time": start_time,
                "first_relative_time": first_time[start_time],
                "capacity_ah": capacity[start_time],
                "n_samples": count[start_time],
            })
    return rows


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    rows = []
    with ZipFile(ARCHIVE) as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith(".csv") and "__MACOSX" not in name
        ]
        for position, member in enumerate(members, 1):
            category = member.split("/")[-2]
            pack = Path(member).stem.removeprefix("battery")
            extracted = extract_member(archive, member)
            for cycle, row in enumerate(extracted):
                rows.append({
                    "category": category,
                    "pack": pack,
                    "cycle": cycle,
                    **row,
                })
            print(
                f"{position}/{len(members)} {category}/{pack}: "
                f"{len(extracted)} reference cycles",
                flush=True,
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "category", "pack", "cycle", "start_time",
            "first_relative_time", "capacity_ah", "n_samples",
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows, {OUT.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
