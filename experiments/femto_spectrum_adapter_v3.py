#!/usr/bin/env python3
"""Versioned FEMTO adapter with absolute and envelope spectra."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from femto_sensor_adapter_v2 import (DT, FS, LEARN, OFFICIAL, RAW, TEST,
                                     acc_index, channel_features, condition,
                                     sampled_files, unit_id)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/femto/features_spectrum_v3.npz"
MANIFEST = ROOT / "data/femto/features_spectrum_v3_manifest.json"
SPECTRUM_BINS = 32
ENVELOPE_BINS = 16


def binned_log_power(signal: np.ndarray, maximum_hz: float, bins: int) -> np.ndarray:
    signal = np.asarray(signal, np.float64)
    centered = signal - signal.mean()
    tapered = centered * np.hanning(len(centered))
    power = np.abs(np.fft.rfft(tapered)) ** 2 / max(len(tapered) ** 2, 1)
    frequency = np.fft.rfftfreq(len(tapered), 1.0 / FS)
    edges = np.linspace(0.0, maximum_hz, bins + 1)
    output = np.empty(bins, np.float32)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (frequency >= left) & (frequency < right)
        output[index] = np.log1p(float(power[selected].mean()) if selected.any() else 0.0)
    return output


def spectral_features(signal: np.ndarray) -> np.ndarray:
    absolute = binned_log_power(signal, FS / 2, SPECTRUM_BINS)
    envelope = np.abs(hilbert(signal - np.mean(signal)))
    envelope_power = binned_log_power(envelope, 2000.0, ENVELOPE_BINS)
    return np.r_[absolute, envelope_power].astype(np.float32)


def build(stride: int = 5) -> dict[str, np.ndarray]:
    rows = []
    audit = []
    for role, names, parent in (("learn", LEARN, "Learning_set"), ("test", TEST, "Test_set")):
        for name in names:
            directory = RAW / parent / name
            all_files = sorted(directory.glob("acc_*.csv"), key=acc_index)
            files = sampled_files(directory, stride)
            first_index = acc_index(all_files[0])
            last_index = acc_index(all_files[-1])
            for path in files:
                array = np.loadtxt(path, delimiter=",", dtype=np.float64)
                if array.ndim != 2 or array.shape[1] != 6:
                    raise ValueError(f"expected 6 columns: {path} has {array.shape}")
                horizontal, vertical = array[:, 4], array[:, 5]
                index = acc_index(path)
                extra = OFFICIAL[name] if role == "test" else 0.0
                statistics = channel_features(horizontal) + channel_features(vertical)
                spectrum = np.r_[spectral_features(horizontal), spectral_features(vertical)]
                rows.append((name, unit_id(name), role, condition(name), index,
                             (index-first_index)*DT, (last_index-index)*DT+extra,
                             statistics, spectrum))
            audit.append({"bearing": name, "role": role, "raw_count": len(all_files),
                          "sampled_count": len(files), "first_index": first_index,
                          "last_index": last_index, "sample_contains_last": files[-1] == all_files[-1]})
            print(name, len(files), flush=True)
    payload = {
        "bearing": np.asarray([r[0] for r in rows]),
        "unit": np.asarray([r[1] for r in rows], np.int32),
        "role": np.asarray([r[2] for r in rows]),
        "condition": np.asarray([r[3] for r in rows], np.float32),
        "recording_index": np.asarray([r[4] for r in rows], np.int32),
        "elapsed_s": np.asarray([r[5] for r in rows], np.float32),
        "y": np.asarray([r[6] for r in rows], np.float32),
        "sensor": np.asarray([r[7] for r in rows], np.float32),
        "spectrum": np.asarray([r[8] for r in rows], np.float32),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    MANIFEST.write_text(json.dumps({"version": 3, "stride": stride,
        "sensor_columns": [4, 5], "forbidden_directory": "Full_Test_Set",
        "spectrum": {"window": "Hann", "power": "absolute log1p mean power",
                     "per_channel_bins": SPECTRUM_BINS, "maximum_hz": FS/2},
        "envelope": {"method": "Hilbert magnitude", "per_channel_bins": ENVELOPE_BINS,
                     "maximum_hz": 2000.0}, "bearings": audit}, indent=2) + "\n")
    return payload


def load(rebuild: bool = False) -> dict[str, np.ndarray]:
    if rebuild or not OUT.exists():
        return build()
    with np.load(OUT) as archive:
        return {key: archive[key] for key in archive.files}


if __name__ == "__main__":
    data = build()
    print(json.dumps({"rows": len(data["y"]), "spectrum_dim": data["spectrum"].shape[1]}, indent=2))
