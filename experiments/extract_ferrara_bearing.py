#!/usr/bin/env python3
"""Extract frozen causal waveform summaries from one Ferrara .7z archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis


ROOT = Path(__file__).resolve().parents[1]


def natural_key(path: Path):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(path: Path) -> np.ndarray:
    payload = loadmat(path, squeeze_me=True)
    if "y" not in payload:
        raise ValueError(f"missing y in {path}")
    y = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if len(y) < 1024:
        raise ValueError(f"too few finite waveform samples in {path}: {len(y)}")
    fs = float(np.asarray(payload.get("Fs", 25600.0)).reshape(-1)[0])
    centered = y - y.mean()
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y * y)))
    std = float(np.std(y))
    ptp = float(np.ptp(y))
    crest = peak / max(rms, 1e-12)
    k = float(kurtosis(y, fisher=False, bias=False))
    spectrum = np.fft.rfft(centered)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) / len(y)
    freq = np.fft.rfftfreq(len(y), d=1.0 / fs)
    bands = []
    for lo, hi in ((0.0, 1000.0), (1000.0, 3000.0), (3000.0, 6000.0), (6000.0, 12800.0)):
        mask = (freq >= lo) & (freq < hi if hi < fs / 2 else freq <= hi)
        bands.append(float(np.log1p(power[mask].mean() if np.any(mask) else 0.0)))
    return np.asarray([peak, rms, std, k, crest, ptp, *bands], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("unit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output or ROOT / "data" / "ferrara_bearing" / "features" / f"{args.unit}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(archive)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {archive}: {digest}")
    with tempfile.TemporaryDirectory(prefix=f"ferrara_{args.unit}_", dir=str(ROOT / "data" / "ferrara_bearing")) as tmp:
        temp = Path(tmp)
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(temp)], check=True)
        files = sorted(temp.rglob("*.mat"), key=natural_key)
        if not files:
            raise RuntimeError(f"no MAT files extracted from {archive}")
        rows = []
        for index, path in enumerate(files):
            rows.append(summarize(path))
            if index % 100 == 0:
                print(args.unit, index, len(files), flush=True)
    matrix = np.vstack(rows)
    names = np.asarray([
        "raw_abs_peak_g", "rms_g", "std_g", "kurtosis", "crest_factor",
        "peak_to_peak_g", "log_band_0_1khz", "log_band_1_3khz",
        "log_band_3_6khz", "log_band_6_12p8khz",
    ])
    np.savez_compressed(output, values=matrix, feature_names=names, unit=args.unit,
                        archive_sha256=digest)
    audit = {
        "unit": args.unit, "archive": archive.name, "archive_sha256": digest,
        "n_acquisitions": int(len(matrix)), "feature_names": names.tolist(),
        "raw_peak_min": float(matrix[:, 0].min()), "raw_peak_max": float(matrix[:, 0].max()),
        "first_20g_index": (int(np.flatnonzero(matrix[:, 0] >= 20.0)[0])
                            if np.any(matrix[:, 0] >= 20.0) else None),
    }
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

