#!/usr/bin/env python3
"""Summarize the pre-holdout structural development path of CCMR v2.3."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ccmr_v23_development_ablation"
ATTEMPTS = [
    ("unshrunk_strong_anchor", "ccmr_v23_cross_domain_development"),
    ("absolute_support_guard", "ccmr_v23_cross_domain_development_v2"),
    ("adaptive_anchor_shadow_gate", "ccmr_v23_cross_domain_development_v3"),
    ("weak_anchor_mass", "ccmr_v23_cross_domain_development_v4"),
    ("dual_adaptive_shadow_gate", "ccmr_v23_cross_domain_development_v5"),
    ("nested_v22_safety_anchor", "ccmr_v23_cross_domain_development_v6"),
    ("five_win_prior_shadow_gate", "ccmr_v23_cross_domain_development_v7"),
    ("condition_evidence_threshold", "ccmr_v23_cross_domain_development_v8"),
    ("contract_specific_threshold_005", "ccmr_v23_cross_domain_development_v9"),
    ("contract_specific_threshold_001", "ccmr_v23_cross_domain_development_v10"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite v2.3 ablation")
    attempts = []
    for label, directory in ATTEMPTS:
        source = ROOT / "results" / directory / "results.json"
        result = json.loads(source.read_text())
        attempts.append({
            "variant": label,
            "artifact": str(source.relative_to(ROOT)),
            **result["summary"],
        })
    final = json.loads((
        ROOT / "results/ccmr_v23_cross_domain_development_v10/results.json"
    ).read_text())
    domain_rows = []
    for name, value in final["nested_loco"].items():
        policy = value["policy"]
        domain_rows.append({
            "domain": name,
            "selected_policy": value["selected_policy"],
            "pooled_rmse": policy["test"]["pooled_rmse"],
            "raw_max_regret": policy["test"]["raw_regret"]["maximum"],
            "false_accept": policy["false_accept"],
        })
    payload = {
        "status": "development-only structural ablation",
        "holdouts_loaded": False,
        "attempts": attempts,
        "final_nested_domains": domain_rows,
        "interpretation": (
            "Nested v2.2 fallback removed catastrophic regret, but the "
            "final LOCO geometric RMSE ratio remained microscopically above 1."
        ),
        "holdout_ablation": (
            "not run because the preregistered promotion gate failed"
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(attempts[-1], indent=2))


if __name__ == "__main__":
    main()
