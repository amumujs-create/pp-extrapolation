#!/usr/bin/env python3
"""Run the official Engression package on the fixed MICH PP split."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / ".benchmark_deps"), str(ROOT.parent / "ca-css-ncmapss")]

from engression_all_positive import run
from extrapolation_competitors_all import datasets

OUT = ROOT / "results/final_engression_extension_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "results.json"
    payload = {
        "protocol": "official engression 0.1.15; validation-only configuration choice; five refits",
        "dataset": "MICH raw-cycle RUL fixed PP split",
    }
    started = time.time()
    payload["mich"] = run(datasets()["mich"])
    payload["seconds"] = time.time() - started
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["mich"]["ensemble"]["pooled"]["r2"], flush=True)


if __name__ == "__main__":
    main()
