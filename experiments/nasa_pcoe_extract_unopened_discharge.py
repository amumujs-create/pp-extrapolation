#!/usr/bin/env python3
"""Extract compact discharge trajectories for unopened NASA PCoE cells."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

from nasa_pcoe_extract_unopened_capacity import ARCHIVE, ROOT, SELECTED

OUT = ROOT / "data/nasa_pcoe_new_unopened/unopened_discharge_trajectories.npz"


def trajectories(payload):
    mat = loadmat(BytesIO(payload), squeeze_me=True, struct_as_record=False)
    key = next(name for name in mat if not name.startswith("__"))
    cycles = np.atleast_1d(mat[key].cycle)
    output = []
    for cycle in cycles:
        if str(getattr(cycle, "type", "")).lower() != "discharge":
            continue
        data = cycle.data
        names = (
            "Time", "Voltage_measured", "Current_measured",
            "Temperature_measured",
        )
        if not all(hasattr(data, name) for name in names):
            continue
        arrays = [
            np.asarray(getattr(data, name), dtype=np.float64).reshape(-1)
            for name in names
        ]
        length = min(map(len, arrays))
        arrays = [array[:length] for array in arrays]
        finite = np.logical_and.reduce([np.isfinite(array) for array in arrays])
        arrays = [array[finite] for array in arrays]
        if len(arrays[0]) < 100:
            continue
        keep = np.r_[True, np.diff(arrays[0]) > 0]
        arrays = [array[keep] for array in arrays]
        if len(arrays[0]) >= 100:
            output.append(arrays)
    return output


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    cell_values, episode_values = [], []
    time_values, voltage_values, current_values, temperature_values = (
        [], [], [], []
    )
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
                    episodes = trajectories(inner.read(member))
                    for episode, arrays in enumerate(episodes):
                        length = len(arrays[0])
                        cell_values.append(np.full(length, cell, dtype="U5"))
                        episode_values.append(np.full(length, episode, dtype=np.int32))
                        time_values.append(arrays[0])
                        voltage_values.append(arrays[1])
                        current_values.append(arrays[2])
                        temperature_values.append(arrays[3])
                    print(cell, len(episodes), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        cell=np.concatenate(cell_values),
        episode=np.concatenate(episode_values),
        time=np.concatenate(time_values),
        voltage=np.concatenate(voltage_values),
        current=np.concatenate(current_values),
        temperature=np.concatenate(temperature_values),
    )
    print(f"wrote {OUT.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
