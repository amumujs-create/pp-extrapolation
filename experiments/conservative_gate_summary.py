#!/usr/bin/env python3
"""Summarize the retrospective unanimous-stability module gate."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    modular = load("results/evidence_gated_modules_v1/results.json")["datasets"]
    hust_stability = load("results/hust_protocol_stability_v1/results.json")
    virkler_stability = load("results/virkler_unit_stability_v1/results.json")
    routes = {
        "hust": (
            "pp"
            if hust_stability["summary"]["pp_stable_protocols"]
            == hust_stability["summary"]["total_protocols"]
            else "plain_nn"
        ),
        "virkler": (
            "pp"
            if virkler_stability["summary"]["unanimous_pp_route"]
            else "plain_nn"
        ),
        "nasa": (
            "pp"
            if all(
                row["selected_arm"] == "pp"
                for row in modular["nasa"]["fold_decisions"]
            )
            else "plain_nn"
        ),
    }
    rows = []
    for dataset, route in routes.items():
        arms = modular[dataset]["arms"]
        selected = float(arms[route]["pooled_r2_mean"])
        oracle_arm = max(arms, key=lambda arm: arms[arm]["pooled_r2_mean"])
        oracle = float(arms[oracle_arm]["pooled_r2_mean"])
        rows.append({
            "dataset": dataset,
            "selected_route": route,
            "selected_pooled_r2": selected,
            "oracle_arm_for_audit_only": oracle_arm,
            "oracle_pooled_r2": oracle,
            "observed_regret": oracle - selected,
        })
    selected_macro = sum(row["selected_pooled_r2"] for row in rows) / len(rows)
    always_pp_macro = sum(
        modular[name]["arms"]["pp"]["pooled_r2_mean"] for name in routes
    ) / len(routes)
    payload = {
        "experiment": "conservative_gate_summary_v1",
        "status": "post-hoc repair after HUST final ranking was observed",
        "rule": "PP only with unanimous source-environment stability; otherwise plain NN",
        "rows": rows,
        "summary": {
            "pp_route_coverage": sum(route == "pp" for route in routes.values()) / len(routes),
            "prediction_coverage": 1.0,
            "dataset_macro_selected_pooled_r2": selected_macro,
            "dataset_macro_always_pp_pooled_r2": always_pp_macro,
            "macro_gain_vs_always_pp": selected_macro - always_pp_macro,
            "worst_dataset_selected_r2": min(row["selected_pooled_r2"] for row in rows),
            "mean_observed_regret": sum(row["observed_regret"] for row in rows) / len(rows),
        },
    }
    output = ROOT / "results/conservative_gate_summary_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Conservative unanimous-stability gate", "",
        "PP is selected only when source environments unanimously pass the stability "
        "gate; otherwise the route falls back to plain NN.", "",
        "| dataset | selected route | selected pooled R² | audit oracle | regret |",
        "|---|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | **{row['selected_route']}** | "
            f"{row['selected_pooled_r2']:.3f} | "
            f"{row['oracle_arm_for_audit_only']} ({row['oracle_pooled_r2']:.3f}) | "
            f"{row['observed_regret']:.3f} |"
        )
    summary = payload["summary"]
    lines += [
        "", f"- PP route coverage: **{summary['pp_route_coverage']:.1%}**",
        f"- prediction coverage: **{summary['prediction_coverage']:.1%}**",
        f"- dataset-macro selected pooled R²: "
        f"**{summary['dataset_macro_selected_pooled_r2']:.3f}**",
        f"- always-PP dataset macro: "
        f"**{summary['dataset_macro_always_pp_pooled_r2']:.3f}**",
        f"- observed gain over always PP: **{summary['macro_gain_vs_always_pp']:+.3f}**",
        f"- worst selected dataset R²: **{summary['worst_dataset_selected_r2']:.3f}**",
        f"- mean observed head regret: **{summary['mean_observed_regret']:.3f}**", "",
        "This threshold was created after the HUST final ranking was known. Zero "
        "observed regret is a retrospective repair result, not validation of the gate.", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
