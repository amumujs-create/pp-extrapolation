#!/usr/bin/env python3
"""Range-extract Stage-2 TP_z RPT capacities (Stage-2 remainder cohort)."""
from __future__ import annotations

import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/multistage_stage2_tpz"
API = "https://api.figshare.com/v2/articles/25975315"
PUBLIC = ROOT / "protocols/MULTISTAGE_STAGE2_TPZ_PUBLIC_MANIFEST.json"
FOLDER = "Multi-Stage_Aging_Study/Stage_2"


def capacity(frame):
    frame = frame.copy()
    frame["run_time"] = pd.to_timedelta(frame["run_time"]).dt.total_seconds()
    for column in ("c_cur", "step_type"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    charge = frame[(frame["step_type"] == 21) & (frame["c_cur"] > 0)].copy()
    discharge = frame[(frame["step_type"] == 22) & (frame["c_cur"] < 0)].copy()
    q_charge = (charge["run_time"].diff() * charge["c_cur"]).fillna(0).cumsum()
    q_discharge = (
        discharge["run_time"].diff() * discharge["c_cur"]
    ).fillna(0).cumsum()
    if len(q_charge) < 2 or len(q_discharge) < 2:
        return float("nan")
    charge_ah = float((q_charge.iloc[-1] - q_charge.iloc[0]) / 3600)
    discharge_ah = float(
        (q_discharge.iloc[0] - q_discharge.iloc[-1]) / 3600
    )
    return (charge_ah + discharge_ah) / 2


def is_rpt(name):
    return bool(
        name.endswith("_ET_T23.csv")
        or name.endswith("_CU.csv")
        or name.endswith("_exCU.csv")
        or name.endswith("_AT_T23.csv")
    )


def process_archive(item):
    rows = []
    with RemoteZip(item["download_url"]) as archive:
        members = sorted(
            (member for member in archive.infolist() if is_rpt(member.filename)),
            key=lambda member: member.filename,
        )
        for member in members:
            match = re.search(
                r"TP_z\d{2}_\d{2}_(\d{2})_",
                Path(member.filename).name,
            )
            if match is None:
                raise ValueError(f"missing RPT sequence: {member.filename}")
            frame = pd.read_csv(io.BytesIO(archive.read(member)))
            rows.append({
                "cell": item["name"].removesuffix(".zip"),
                "archive_id": item["id"],
                "archive_md5": item["supplied_md5"],
                "rpt_sequence": int(match.group(1)),
                "member": member.filename,
                "member_crc": member.CRC,
                "capacity_ah": capacity(frame),
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "rpt_capacity.csv"
    if output.exists():
        raise RuntimeError("refusing to overwrite Stage-2 TP_z extraction")
    public = json.loads(PUBLIC.read_text())
    metadata = requests.get(API, timeout=60).json()
    folder = metadata["folder_structure"]
    live = [
        item for item in metadata["files"]
        if folder.get(str(item["id"])) == FOLDER
        and item["name"].startswith("TP_z")
        and item["name"].endswith(".zip")
    ]
    live = sorted(live, key=lambda item: item["name"])
    expected = public["archives"]
    if len(live) != len(expected):
        raise RuntimeError(
            f"expected {len(expected)} TP_z archives, found {len(live)}"
        )
    for left, right in zip(live, expected):
        for key in ("id", "name", "size", "supplied_md5"):
            if left[key] != right[key]:
                raise RuntimeError(
                    f"manifest mismatch on {right['name']}:{key}"
                )
    selected = [{
        key: item[key]
        for key in ("id", "name", "size", "download_url", "supplied_md5")
    } for item in live]
    (OUT / "source_manifest.json").write_text(json.dumps({
        "doi": public["doi"],
        "article_id": public["article_id"],
        "folder": FOLDER,
        "population_rule": public["population_rule"],
        "archives": selected,
    }, indent=2) + "\n")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_archive, item): item for item in selected
        }
        for number, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            rows.extend(future.result())
            print(f"{number}/{len(selected)} {item['name']}", flush=True)
    result = pd.DataFrame(rows).sort_values(["cell", "rpt_sequence"])
    result.to_csv(output, index=False)
    print(json.dumps({
        "cells": int(result["cell"].nunique()),
        "rows": len(result),
        "finite": int(result["capacity_ah"].notna().sum()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }), flush=True)


if __name__ == "__main__":
    main()
