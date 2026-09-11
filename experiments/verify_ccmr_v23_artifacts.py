#!/usr/bin/env python3
"""Verify the rejected v2.3 hash manifest and holdout firewall."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocols/CCMR_V23_REJECTED_MANIFEST.json"


def main():
    manifest = json.loads(MANIFEST.read_text())
    failures = []
    for _, (relative, expected) in manifest["artifacts"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if observed != expected:
            failures.append(relative)
    frozen = ROOT / "protocols/CCMR_V23_FROZEN_MANIFEST.json"
    replay = ROOT / "results/ccmr_v23_frozen_holdout_replay/results.json"
    if frozen.exists():
        failures.append(str(frozen.relative_to(ROOT)))
    if replay.exists():
        failures.append(str(replay.relative_to(ROOT)))
    if failures:
        raise RuntimeError(f"v2.3 artifact verification failed: {failures}")
    print(json.dumps({
        "verified_hashes": len(manifest["artifacts"]),
        "holdout_replay_authorized": False,
        "holdout_replay_absent": True,
    }, indent=2))


if __name__ == "__main__":
    main()
