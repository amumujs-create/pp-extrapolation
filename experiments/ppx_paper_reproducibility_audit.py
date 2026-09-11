#!/usr/bin/env python3
"""Reproducibility, evidence-tier, and selection-firewall audit for PP-X."""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.executor_policy import select_residual_executor  # noqa: E402
from pp_extrapolation.paper_ppx import select_paper_ppx  # noqa: E402
from pp_extrapolation.transferability_gate import select_ppx_route  # noqa: E402

OUT = ROOT / "results/ppx_paper_reproducibility_audit_v1"
PROTOCOL = ROOT / "protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md"
REGISTRY = ROOT / "results/ppx_unified_final_v1/registry.json"
EVIDENCE = ROOT / "results/final_modular_pp_evidence_v1/results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature_audit(function):
    names = tuple(inspect.signature(function).parameters)
    forbidden = tuple(
        name for name in names
        if "test" in name.lower() or name.lower() in {"y_test", "test_y"}
    )
    return {
        "function": f"{function.__module__}.{function.__name__}",
        "parameters": names,
        "forbidden_test_parameters": forbidden,
        "passed": not forbidden,
    }


def git_introduction(path: Path):
    relative = str(path.relative_to(ROOT))
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    commits = [line for line in result.stdout.splitlines() if line]
    return commits[-1] if commits else None


def main():
    registry = json.loads(REGISTRY.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    routes = []
    for route in registry["routes"]:
        artifact = ROOT / route["artifact"]
        routes.append({
            "dataset": route["dataset"],
            "route": route["route"],
            "artifact": route["artifact"],
            "artifact_exists": artifact.exists(),
            "artifact_sha256": sha256(artifact) if artifact.exists() else None,
            "tier": (
                "retrospective_limitation"
                if route["dataset"] in {"XJTU", "FEMTO", "NASA milling"}
                else "retrospective_main"
            ),
        })
    signature_rows = [
        signature_audit(select_ppx_route),
        signature_audit(select_residual_executor),
        signature_audit(select_paper_ppx),
    ]
    aggregate = evidence["aggregate"]
    payload = {
        "status": "PP-X paper reproducibility audit complete",
        "owner": "박진서",
        "paper_method": "PP-X",
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": sha256(PROTOCOL),
            "introduction_commit": git_introduction(PROTOCOL),
        },
        "selection_firewall": {
            "selectors": signature_rows,
            "all_passed": all(row["passed"] for row in signature_rows),
            "note": (
                "Signature exclusion is necessary but not sufficient; "
                "integration tests also verify fallback and contract behavior."
            ),
        },
        "registry": {
            "routes": len(routes),
            "artifacts_present": sum(row["artifact_exists"] for row in routes),
            "retrospective_main": sum(
                row["tier"] == "retrospective_main" for row in routes
            ),
            "retrospective_limitation": sum(
                row["tier"] == "retrospective_limitation" for row in routes
            ),
            "prospective_confirmations": 0,
            "rows": routes,
        },
        "corrected_paired_evidence": {
            "datasets": aggregate["datasets"],
            "physical_units": aggregate["total_physical_units"],
            "dataset_wins": aggregate["pp_ensemble_wins"],
            "dataset_sign_test_p": aggregate["dataset_sign_test_two_sided"],
            "equal_dataset_mean_log_rmse_ratio": (
                aggregate["equal_dataset_mean_unit_log_rmse_ratio"]
            ),
            "hierarchical_log_ratio_ci95": (
                aggregate["hierarchical_log_ratio_ci95"]
            ),
            "geometric_mean_rmse_reduction": (
                aggregate["geometric_mean_rmse_reduction"]
            ),
        },
        "hard_limitations": [
            "The 9-domain main evidence is retrospective and development-final.",
            "The 12 routes are heterogeneous adapters, not one shared fitted neural network.",
            "A simple validation-RMSE route has four false accepts in 12 domains.",
            "The frozen paper policy still requires a genuinely unopened prospective cohort.",
            "Strongest pooled comparators and row-aligned paired comparators differ on some domains.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "protocol_commit": payload["protocol"]["introduction_commit"],
        "selection_firewall": payload["selection_firewall"]["all_passed"],
        "registry": payload["registry"],
        "paired": payload["corrected_paired_evidence"],
    }, indent=2))


if __name__ == "__main__":
    main()
