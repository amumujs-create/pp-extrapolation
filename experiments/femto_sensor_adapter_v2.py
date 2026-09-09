#!/usr/bin/env python3
"""Audited PHM2012/FEMTO adapter using the two acceleration columns.

The public files contain four metadata/time columns followed by horizontal and
vertical acceleration.  Legacy PP experiments accidentally read columns 0/1.
This adapter is deliberately versioned and never reads Full_Test_Set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/femto/raw"
OUT = ROOT / "data/femto/features_sensor_v2.npz"
MANIFEST = ROOT / "data/femto/features_sensor_v2_manifest.json"
FS = 25600.0
DT = 10.0
LEARN = ("Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2")
TEST = ("Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3")
OFFICIAL = {"Bearing1_3":5730,"Bearing1_4":339,"Bearing1_5":1610,"Bearing1_6":1460,
            "Bearing1_7":7570,"Bearing2_3":7530,"Bearing2_4":1390,"Bearing2_5":3090,
            "Bearing2_6":1290,"Bearing2_7":580,"Bearing3_3":820}
BANDS = ((0,1000),(1000,5000),(5000,12000))


def acc_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def sampled_files(directory: Path, stride: int = 5) -> list[Path]:
    files = sorted(directory.glob("acc_*.csv"), key=acc_index)
    if not files:
        raise FileNotFoundError(directory)
    chosen = files[::stride]
    if chosen[-1] != files[-1]:
        chosen.append(files[-1])
    return chosen


def kurtosis(x: np.ndarray) -> float:
    s = x.std()
    return float(np.mean(((x-x.mean())/s)**4)-3) if s > 1e-12 else 0.0


def channel_features(x: np.ndarray) -> list[float]:
    centered = x-x.mean()
    spec = np.abs(np.fft.rfft(centered))
    freq = np.fft.rfftfreq(x.size, 1/FS)
    bands = [float(spec[(freq>=lo)&(freq<hi)].mean()) for lo,hi in BANDS]
    rms = float(np.sqrt(np.mean(x*x)))
    return [rms, float(x.std()), kurtosis(x), float(np.max(np.abs(x))/(rms+1e-12)), *bands]


def condition(name: str) -> int:
    return int(name[7])


def unit_id(name: str) -> int:
    a,b = name.replace("Bearing", "").split("_")
    return int(a)*10+int(b)


def build(stride: int = 5) -> dict[str, np.ndarray]:
    rows = []
    audit = []
    for role, names, parent in (("learn", LEARN, "Learning_set"), ("test", TEST, "Test_set")):
        for name in names:
            directory = RAW / parent / name
            all_files = sorted(directory.glob("acc_*.csv"), key=acc_index)
            files = sampled_files(directory, stride)
            last_idx = acc_index(all_files[-1])
            for path in files:
                arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 6:
                    raise ValueError(f"expected 6 columns: {path} has {arr.shape}")
                # PHM2012 layout: hour, minute, second, microsecond, h_acc, v_acc.
                h,v = arr[:,4], arr[:,5]
                idx = acc_index(path)
                extra = OFFICIAL[name] if role == "test" else 0.0
                rows.append((name, unit_id(name), role, condition(name), idx,
                             (idx-acc_index(all_files[0]))*DT,
                             (last_idx-idx)*DT+extra, *(channel_features(h)+channel_features(v))))
            audit.append({"bearing":name,"role":role,"raw_count":len(all_files),
                          "sampled_count":len(files),"first_index":acc_index(all_files[0]),
                          "last_index":last_idx,"sample_contains_last":files[-1]==all_files[-1],
                          "last_sha256":hashlib.sha256(all_files[-1].read_bytes()).hexdigest()})
    a = np.asarray(rows, dtype=object)
    payload = {"bearing":a[:,0].astype(str),"unit":a[:,1].astype(np.int32),
               "role":a[:,2].astype(str),"condition":a[:,3].astype(np.float32),
               "recording_index":a[:,4].astype(np.int32),"elapsed_s":a[:,5].astype(np.float32),
               "y":a[:,6].astype(np.float32),"sensor":a[:,7:].astype(np.float32)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    MANIFEST.write_text(json.dumps({"version":2,"stride":stride,"sensor_columns":[4,5],
        "forbidden_directory":"Full_Test_Set","feature_order_per_channel":["rms","std","kurtosis","crest","fft_0_1k","fft_1_5k","fft_5_12k"],
        "bearings":audit}, indent=2)+"\n")
    return payload


def load(rebuild: bool = False) -> dict[str, np.ndarray]:
    if rebuild or not OUT.exists():
        return build()
    with np.load(OUT) as z:
        return {k:z[k] for k in z.files}


if __name__ == "__main__":
    p = build()
    print(json.dumps({"rows":len(p["y"]),"sensor_dim":p["sensor"].shape[1],
                      "test_endpoints":int(sum((p["role"]=="test") & np.r_[p["bearing"][1:]!=p["bearing"][:-1], True]))}, indent=2))
