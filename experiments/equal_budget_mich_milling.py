#!/usr/bin/env python3
"""Extend the locked 29-candidate competitor protocol to MICH and Milling.

This is intentionally a new result directory: older short-grid results remain
historical and are never overwritten or pooled with this audit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]

from equal_budget_competitors import KINDS, tune
from extrapolation_competitors_all import datasets

OUT = ROOT / "results/final_equal_budget_extension_v1"


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "protocol": "29 validation candidates per method; selected configuration refit over seeds 42--46",
        "scope": "MICH and NASA Milling extension; separate from historical short-grid runs",
        "datasets": {},
    }
    available = datasets()
    for name in ("mich", "milling"):
        parts = available[name]
        # Milling has one validation group; the output records this limitation
        # rather than pretending it supplies a stable model-ranking signal.
        result["datasets"].setdefault(name, {"validation_groups": int(len(set(parts[1]["groups"])))})
        for kind in KINDS:
            if kind in result["datasets"][name]:
                continue
            started = time.time()
            result["datasets"][name][kind] = tune(name, kind, parts)
            result["datasets"][name][kind]["seconds"] = time.time() - started
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(name, kind, result["datasets"][name][kind]["ensemble"]["pooled"]["r2"], flush=True)


if __name__ == "__main__":
    main()
