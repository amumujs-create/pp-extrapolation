#!/usr/bin/env python3
"""Freeze promoted CCMR v2.3 sources before holdout loaders are imported."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = "results/ccmr_v23_cross_domain_development_v10/results.json"
MANIFEST = ROOT / "protocols/CCMR_V23_FROZEN_MANIFEST.json"
REJECTED = ROOT / "protocols/CCMR_V23_REJECTED_MANIFEST.json"
ARTIFACTS = {
    "firewall": "protocols/CCMR_V23_AC_CRPE_FIREWALL.md",
    "contracts": "src/pp_extrapolation/contracts.py",
    "predictor_portfolio": "src/pp_extrapolation/predictor_portfolio.py",
    "contract_risk_experts": "src/pp_extrapolation/contract_risk_experts.py",
    "causal_dynamics_bank": "src/pp_extrapolation/causal_dynamics_bank.py",
    "development_runner": "experiments/ccmr_v23_cross_domain_development.py",
    "development_result": DEVELOPMENT,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if MANIFEST.exists():
        raise RuntimeError("refusing to overwrite v2.3 frozen manifest")
    result = json.loads((ROOT / DEVELOPMENT).read_text())
    if not result["summary"]["promoted"]:
        payload = {
            "model": "CCMR v2.3 AC-CRPE",
            "owner": "박진서",
            "status": "rejected by preregistered promotion gate",
            "holdout_replay_authorized": False,
            "holdouts_loaded_by_freezer": False,
            "artifacts": {
                name: [relative, digest(ROOT / relative)]
                for name, relative in ARTIFACTS.items()
            },
            "development_summary": result["summary"],
        }
        if REJECTED.exists():
            raise RuntimeError("refusing to overwrite rejected manifest")
        REJECTED.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({
            "manifest": str(REJECTED.relative_to(ROOT)),
            "holdout_replay_authorized": False,
        }, indent=2))
        return
    payload = {
        "model": "CCMR v2.3 AC-CRPE",
        "owner": "박진서",
        "promotion_gate_passed": True,
        "frozen_before_holdout_replay": True,
        "holdouts_loaded_by_freezer": False,
        "global_policy": result["global_policy_for_frozen_replay"],
        "artifacts": {
            name: [relative, digest(ROOT / relative)]
            for name, relative in ARTIFACTS.items()
        },
        "development_summary": result["summary"],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "global_policy": payload["global_policy"],
        "artifacts": len(payload["artifacts"]),
    }, indent=2))


if __name__ == "__main__":
    main()
