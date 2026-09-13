#!/usr/bin/env python3
"""Prior-off fallback portfolio audit on DS03 (and optional NASA note).

Question
--------
When PP-X rejects prior routes, can validation choose a stronger fallback from
{direct NN, Engression, FT-Transformer, GroupDRO, ...} without peeking at test?

This audit reuses frozen equal-budget artifacts and the frozen PP-X DS03
selection. It does not retune Algorithm 1's prior-on core.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EQUAL = ROOT / "results/ncmapss_ds03_equal_budget_v1/results.json"
PPX_REVEAL = ROOT / "results/ncmapss_ds03_ppx_v1/reveal.json"
PPX_SELECTION = ROOT / "results/ncmapss_ds03_ppx_v1/selection.json"
OUT = ROOT / "results/ds03_prior_off_fallback_portfolio_v1"

PORTFOLIO = (
    "plain_mlp",
    "engression",
    "ft_transformer",
    "groupdro",
    "linear_tail_rbf",
    "vrex",
    "monotone",
    "svgp",
)


def selected_validation_mse(row: dict) -> float:
    search = row["search"]
    chosen = row["selected_config"]
    matches = [item["validation_mse"] for item in search if item["config"] == chosen]
    if matches:
        return float(matches[0])
    return float(min(item["validation_mse"] for item in search))


def main() -> None:
    equal = json.loads(EQUAL.read_text())["models"]
    selection = json.loads(PPX_SELECTION.read_text())
    reveal = json.loads(PPX_REVEAL.read_text())

    ppx_route = selection["selected_route"]
    val_rows = selection["validation_evidence"]
    ppx_val = float(next(item["validation_loss"] for item in val_rows if item["executor"] == ppx_route))
    ppx_test_r2 = float(reveal["pooled_r2"])

    rows = []
    for name in PORTFOLIO:
        model = equal[name]
        val = selected_validation_mse(model)
        test_r2 = float(model["ensemble"]["pooled"]["r2"])
        rows.append({
            "model": name,
            "validation_mse": val,
            "test_pooled_r2": test_r2,
            "beats_ppx_fallback_on_test": test_r2 > ppx_test_r2 + 1e-12,
        })

    # Also score the frozen PP-X direct fallback as a selectable arm.
    portfolio_with_ppx = rows + [{
        "model": "ppx_direct_fallback_frozen",
        "validation_mse": ppx_val,
        "test_pooled_r2": ppx_test_r2,
        "beats_ppx_fallback_on_test": False,
    }]

    ranking = sorted(
        portfolio_with_ppx,
        key=lambda item: (item["validation_mse"], -item["test_pooled_r2"]),
    )
    chosen = ranking[0]
    engression = next(item for item in rows if item["model"] == "engression")
    oracle = max(portfolio_with_ppx, key=lambda item: item["test_pooled_r2"])

    # Restricted portfolio requested in discussion: {direct NN, Engression, ...}
    # Minimal set: frozen direct + Engression + FT.
    restricted = [
        next(item for item in portfolio_with_ppx if item["model"] == "ppx_direct_fallback_frozen"),
        engression,
        next(item for item in rows if item["model"] == "ft_transformer"),
        next(item for item in rows if item["model"] == "plain_mlp"),
    ]
    restricted_chosen = sorted(
        restricted,
        key=lambda item: (item["validation_mse"], -item["test_pooled_r2"]),
    )[0]

    payload = {
        "scope": (
            "DS03 prior-off fallback portfolio only. Prior-on PP-X routes remain "
            "rejected by the frozen paper gate and are not retuned."
        ),
        "frozen_ppx": {
            "selected_route": ppx_route,
            "validation_mse": ppx_val,
            "test_pooled_r2": ppx_test_r2,
            "reason": selection.get("reason"),
            "prior_admissibility": selection.get("prior_admissibility_evidence"),
        },
        "portfolio": portfolio_with_ppx,
        "validation_selected_fallback": chosen,
        "restricted_portfolio_selected": restricted_chosen,
        "test_oracle_fallback": {
            "model": oracle["model"],
            "test_pooled_r2": oracle["test_pooled_r2"],
            "validation_mse": oracle["validation_mse"],
        },
        "engression": engression,
        "findings": {
            "validation_picks_engression": chosen["model"] == "engression",
            "restricted_picks_engression": restricted_chosen["model"] == "engression",
            "validation_selected_model": chosen["model"],
            "validation_selected_test_r2": chosen["test_pooled_r2"],
            "restricted_selected_model": restricted_chosen["model"],
            "restricted_selected_test_r2": restricted_chosen["test_pooled_r2"],
            "delta_vs_ppx_fallback": chosen["test_pooled_r2"] - ppx_test_r2,
            "restricted_delta_vs_ppx_fallback": (
                restricted_chosen["test_pooled_r2"] - ppx_test_r2
            ),
            "engression_validation_worse_than_ppx_direct": (
                engression["validation_mse"] > ppx_val
            ),
            "would_test_oracle_need_engression": oracle["model"] == "engression",
            "interpretation": (
                "On DS03, Engression is best on test but worse than the frozen "
                "PP-X direct fallback on validation MSE. A validation-only "
                "portfolio therefore does not select Engression; full portfolio "
                f"selects {chosen['model']}, restricted set selects "
                f"{restricted_chosen['model']}."
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["findings"], indent=2))
    print("validation_selected", chosen)
    print("ppx_direct", ppx_val, ppx_test_r2)
    print("engression", engression["validation_mse"], engression["test_pooled_r2"])


if __name__ == "__main__":
    main()
