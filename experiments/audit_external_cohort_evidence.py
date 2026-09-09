#!/usr/bin/env python3
"""Audit the evidentiary status of external PP cohort artifacts.

This script does not train or select a model.  It reads immutable result artifacts
and separates predictor performance from the much stronger claim that a
pre-outcome routing certificate was prospectively validated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARTIFACTS = {
    "HNEI locked split": {
        "path": "hnei_external_locked_v1/results.json",
        "kind": "external locked split",
        "expected": "developmental_repair",
    },
    "MATWI tool wear": {
        "path": "matwi_tool_life_untouched_v1/results.json",
        "kind": "untouched external trajectory",
        "expected": "confirmatory",
    },
    "Na-ion 80%-EOL": {
        "path": "naion_80eol_prospective_gate/results.json",
        "kind": "prospective gate",
        "expected": "prospective_gate",
    },
    "Zn-ion RBF v2": {
        "path": "znion_rbf_regime_confirmation_v2/results.json",
        "kind": "untouched external trajectory",
        "expected": "inconclusive_small_n",
    },
    "Zn-ion RBF v3": {
        "path": "znion_rbf_regime_confirmation_v3/results.json",
        "kind": "untouched external trajectory",
        "expected": "inconclusive_small_n",
    },
    "XJTU transfer": {
        "path": "xjtu_untouched_v1/results.json",
        "kind": "untouched operating-condition transfer",
        "expected": "failed",
    },
    "FEMTO prospective": {
        "path": "femto_pp_prospective_v1/results.json",
        "kind": "prospective bearing tail",
        "expected": "invalidated_loader",
    },
}


def metric(obj: object) -> float | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("r2"), (int, float)):
        return float(obj["r2"])
    pooled = obj.get("pooled")
    if isinstance(pooled, dict) and isinstance(pooled.get("r2"), (int, float)):
        return float(pooled["r2"])
    return None


def first_metric(data: dict, names: list[str]) -> float | None:
    for name in names:
        value = metric(data.get(name))
        if value is not None:
            return value
    return None


def cohort_metrics(data: dict) -> tuple[float | None, float | None]:
    """Handle the two historical artifact schemas without silently dropping a result."""
    pp_r2 = first_metric(data, ["pp_ensemble", "pp"])
    mlp_r2 = first_metric(data, ["plain_ensemble", "plain_mlp", "plain_nn"])
    projected = data.get("projected_all")
    if isinstance(projected, dict):
        pp_r2 = pp_r2 if pp_r2 is not None else metric(projected.get("pp"))
        mlp_r2 = mlp_r2 if mlp_r2 is not None else metric(projected.get("plain"))
    return pp_r2, mlp_r2


def verdict(spec: dict, data: dict, pp_r2: float | None, mlp_r2: float | None) -> str:
    expected = spec["expected"]
    if expected == "developmental_repair":
        return "developmental only: post-outcome normalization repair prevents prospective claim"
    if expected == "invalidated_loader":
        return "invalidated for final FEMTO claim: causal input loader/caching repair is required"
    if expected == "inconclusive_small_n":
        return "inconclusive: insufficient independent eligible units despite reported point estimate"
    if expected == "prospective_gate":
        return "gate failure unless the artifact explicitly records a predeclared successful certificate"
    if data.get("confirmatory_success") is True:
        return "confirmatory success"
    if pp_r2 is not None and mlp_r2 is not None and pp_r2 > mlp_r2:
        return "predictor wins, but artifact does not meet declared confirmatory-success criterion"
    return "not a PP success"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/cohort_evidence_audit_v2"))
    args = parser.parse_args()
    rows = []
    for cohort, spec in ARTIFACTS.items():
        source = args.results_root / spec["path"]
        row = {"cohort": cohort, "artifact": str(source), "kind": spec["kind"]}
        if not source.exists():
            row.update({"available": False, "verdict": "artifact missing"})
        else:
            data = json.loads(source.read_text())
            pp_r2, mlp_r2 = cohort_metrics(data)
            row.update({
                "available": True,
                "status": data.get("status"),
                "pp_pooled_r2": pp_r2,
                "mlp_pooled_r2": mlp_r2,
                "gain_r2": None if pp_r2 is None or mlp_r2 is None else pp_r2 - mlp_r2,
                "certificate_approved": data.get("predeclared_certificate_approved", data.get("gate", {}).get("global_approved") if isinstance(data.get("gate"), dict) else None),
                "confirmatory_success": data.get("confirmatory_success", data.get("prospective_gate_success")),
                "verdict": verdict(spec, data, pp_r2, mlp_r2),
            })
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    header = "| cohort | evidence type | PP pooled R² | MLP pooled R² | ΔR² | final evidentiary status |\n|---|---|---:|---:|---:|---|\n"
    lines = [header]
    for row in rows:
        f = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['cohort']} | {row['kind']} | {f(row.get('pp_pooled_r2'))} | "
            f"{f(row.get('mlp_pooled_r2'))} | {f(row.get('gain_r2'))} | {row['verdict']} |\n"
        )
    lines.extend([
        "\n## Interpretation\n",
        "This is an artifact audit, not a new experiment. A predictor win, an untouched split, and a successful pre-outcome certificate are different claims. Only the last supports deployment-gate validation.\n",
        "A result marked developmental, invalidated, or inconclusive must not be pooled with confirmatory successes in the PP paper.\n",
    ])
    (args.output / "RESULTS_KO.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
