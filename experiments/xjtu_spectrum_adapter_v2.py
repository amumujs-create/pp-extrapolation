#!/usr/bin/env python3
"""Audited XJTU-SY recording-level vibration spectrum cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from xjtu_untouched import CONDITIONS, DATA, recording_stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/xjtu_sy/features_spectrum_v2.npz"
MANIFEST = ROOT / "data/xjtu_sy/features_spectrum_v2_manifest.json"
FS = 25600.0


def log_power_bins(signal: np.ndarray, maximum_hz: float, bins: int) -> np.ndarray:
    centered = np.asarray(signal, np.float64) - np.mean(signal)
    tapered = centered * np.hanning(len(centered))
    power = np.abs(np.fft.rfft(tapered))**2 / max(len(tapered)**2, 1)
    frequency = np.fft.rfftfreq(len(tapered), 1/FS)
    edges = np.linspace(0, maximum_hz, bins+1)
    return np.asarray([np.log1p(float(power[(frequency>=a)&(frequency<b)].mean()))
                       for a,b in zip(edges[:-1], edges[1:])], np.float32)


def spectrum(signal: np.ndarray) -> np.ndarray:
    absolute = log_power_bins(signal, FS/2, 32)
    envelope = np.abs(hilbert(signal-np.mean(signal)))
    return np.r_[absolute, log_power_bins(envelope, 2000, 16)].astype(np.float32)


def build() -> dict[str, np.ndarray]:
    stats=[]; spectra=[]; units=[]; conditions=[]; positions=[]; lives=[]; audit=[]
    for condition in CONDITIONS:
        for folder in sorted((DATA/condition).glob("Bearing*")):
            files=sorted(folder.glob("*.csv"),key=lambda path:int(path.stem)); life=len(files)
            for position,path in enumerate(files,1):
                array=np.loadtxt(path,delimiter=",",skiprows=1,dtype=np.float64)
                if array.shape!=(32768,2): raise ValueError(f"unexpected shape {array.shape}: {path}")
                stats.append(recording_stats(path)); spectra.append(np.r_[spectrum(array[:,0]),spectrum(array[:,1])])
                units.append(folder.name);conditions.append(condition);positions.append(position);lives.append(life)
            audit.append({"condition":condition,"unit":folder.name,"recordings":life,
                          "first_file":files[0].name,"last_file":files[-1].name})
            print(condition,folder.name,life,flush=True)
    payload={"stats":np.asarray(stats,np.float32),"spectrum":np.asarray(spectra,np.float32),
             "units":np.asarray(units),"conditions":np.asarray(conditions),
             "positions":np.asarray(positions,np.int32),"lives":np.asarray(lives,np.int32)}
    np.savez_compressed(OUT,**payload)
    MANIFEST.write_text(json.dumps({"version":2,"sampling_hz":FS,"rows":len(units),
        "spectrum":"32 absolute log-power + 16 Hilbert-envelope log-power bins per channel",
        "audit":audit},indent=2)+"\n")
    return payload


def load(rebuild=False):
    if rebuild or not OUT.exists(): return build()
    with np.load(OUT) as archive:return {key:archive[key] for key in archive.files}


if __name__=="__main__":
    value=build();print(json.dumps({"rows":len(value["units"]),"spectrum_dim":value["spectrum"].shape[1]},indent=2))
