#!/usr/bin/env python3
"""Promotion audit: soft prior-gate (Policy A) vs frozen hard reject.

When stage-1 prior admissibility fails, frozen PP-X never scores
prior+residual executors. Policy A still runs frozen stage-2 unit-risk
approval. This retrospective counterfactual uses the 12-setting
common-backbone archive. Decisions use validation only; test labels score
false accept / reject after the fact.

See protocols/PPX_PRIOR_GATE_SOFTENING_A_PROTOCOL.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pp_extrapolation.paper_ppx import (  # noqa: E402
    PPXCandidateEvidence,
    PPXContract,
    select_paper_ppx,
)
from pp_extrapolation.presets import paper_ppx_approval_thresholds  # noqa: E402
from pp_extrapolation.transferability_gate import PriorEvidence  # noqa: E402
from ppx_paper_policy_audit import test_effect, validation_evidence  # noqa: E402

SOURCE = ROOT / "results/cross_domain_mechanism_v2"
OUT = ROOT / "results/ppx_prior_gate_softening_a_v1"
OWNER = "박진서"


def stage2_accepts_pp(validation: dict) -> tuple[bool, str]:
    thr = paper_ppx_approval_thresholds()
    contract = PPXContract(
        known_boundary=True,
        ordered_progression=True,
        causal_history=False,
        observed_regime=False,
        support_heterogeneity_available=False,
        fallback="direct_fallback",
    )
    # Hold prior admissible so only stage-2 criteria bind (Policy A path).
    prior = PriorEvidence(
        known_boundary=True,
        complete_groups=0,
        minimum_complete_groups_per_regime=0,
    )
    decision = select_paper_ppx(
        contract,
        prior,
        (
            PPXCandidateEvidence(
                "direct_fallback",
                validation["direct_loss"],
                0.0,
                1.0,
            ),
            PPXCandidateEvidence(
                "unbounded",
                validation["pp_loss"],
                validation["unit_win_fraction"],
                validation["worst_unit_ratio"],
                unit_gain_ci_low=float("-inf"),
            ),
        ),
        min_relative_improvement=thr["min_relative_improvement"],
        min_unit_win_fraction=thr["min_unit_win_fraction"],
        max_worst_unit_rmse_ratio=thr["max_worst_unit_rmse_ratio"],
        min_unit_gain_ci_low=float("-inf"),
    )
    return decision.executor == "unbounded", decision.reason


def score(selected: np.ndarray, actual: np.ndarray, effects: np.ndarray) -> dict:
    selected = np.asarray(selected, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    effects = np.asarray(effects, float)
    deployed = np.where(selected, effects, 0.0)
    return {
        "accepted": int(selected.sum()),
        "correct": int(np.sum(selected == actual)),
        "accuracy": float(np.mean(selected == actual)),
        "false_accepts": int(np.sum(selected & ~actual)),
        "false_rejects": int(np.sum(~selected & actual)),
        "equal_dataset_mean_deployed_log_rmse_ratio": float(np.mean(deployed)),
        "oracle_positive": int(actual.sum()),
    }


def promotion_verdict(
    a: dict,
    frozen_fail: dict,
    operational: dict,
    *,
    n_true_rescues: int,
    n_false_rescues: int,
) -> dict:
    weak = {
        "fr_improves_vs_frozen_s1_fail": a["false_rejects"]
        < frozen_fail["false_rejects"],
        "fa_not_worse_than_operational": a["false_accepts"]
        <= operational["false_accepts"],
        "deployed_effect_improves_vs_frozen_s1_fail": (
            a["equal_dataset_mean_deployed_log_rmse_ratio"]
            > frozen_fail["equal_dataset_mean_deployed_log_rmse_ratio"]
        ),
    }
    strict = {
        "has_true_rescue": n_true_rescues >= 1,
        "zero_false_rescue": n_false_rescues == 0,
        "deployed_effect_improves_vs_frozen_s1_fail": weak[
            "deployed_effect_improves_vs_frozen_s1_fail"
        ],
    }
    weak_pass = all(weak.values())
    strict_pass = all(strict.values())
    return {
        "promote_policy_A": strict_pass,
        "weak_evidence_only": bool(weak_pass and not strict_pass),
        "strict_criteria": strict,
        "weak_criteria": weak,
        "n_true_rescues": n_true_rescues,
        "n_false_rescues": n_false_rescues,
        "decision": (
            "promote_candidate"
            if strict_pass
            else "reject_policy_A_keep_frozen_hard_s1"
        ),
        "rationale": (
            "Strict promotion criteria passed."
            if strict_pass
            else (
                "Weak evidence only: A beats always-direct under forced S1 "
                "failure, but re-opens false rescues (or lacks a clean "
                "true-only rescue). Keep hard stage-1 reject."
                if weak_pass
                else (
                    "Policy A fails even the weak criteria versus "
                    "always-direct / operational FA; keep hard stage-1 reject."
                )
            )
        ),
    }


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    rows = []
    actual = []
    effects = []
    stage2 = []
    for name in metadata["datasets"]:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        validation = validation_evidence(saved)
        test = test_effect(saved)
        accepts, reason = stage2_accepts_pp(validation)
        helped = bool(test["pp_helped"])
        effect = float(test["mean_unit_log_rmse_ratio"])
        rows.append(
            {
                "dataset": name,
                "validation": {
                    "relative_gain": 1.0
                    - validation["pp_loss"] / validation["direct_loss"],
                    "unit_win_fraction": validation["unit_win_fraction"],
                    "worst_unit_ratio": validation["worst_unit_ratio"],
                },
                "stage2_accepts_pp": accepts,
                "stage2_reason": reason,
                "test_pp_helped": helped,
                "test_mean_unit_log_rmse_ratio": effect,
                # Under forced S1 fail:
                "frozen_s1_fail_accepts_pp": False,
                "policy_A_s1_fail_accepts_pp": accepts,
                # Reference: current operational path (S1 held true).
                "operational_s1_pass_accepts_pp": accepts,
            }
        )
        actual.append(helped)
        effects.append(effect)
        stage2.append(accepts)

    actual_arr = np.asarray(actual, dtype=bool)
    effects_arr = np.asarray(effects, float)
    stage2_arr = np.asarray(stage2, dtype=bool)
    always_direct = np.zeros_like(stage2_arr)

    summaries = {
        "frozen_s1_fail": score(always_direct, actual_arr, effects_arr),
        "policy_A_s1_fail": score(stage2_arr, actual_arr, effects_arr),
        "operational_s1_pass": score(stage2_arr, actual_arr, effects_arr),
    }

    # Where A would diverge from hard S1 fail: stage2 accepts while S1 blocked.
    rescue_rows = [
        row
        for row in rows
        if row["policy_A_s1_fail_accepts_pp"] and not row["frozen_s1_fail_accepts_pp"]
    ]
    rescue_true = [row for row in rescue_rows if row["test_pp_helped"]]
    rescue_false = [row for row in rescue_rows if not row["test_pp_helped"]]
    verdict = promotion_verdict(
        summaries["policy_A_s1_fail"],
        summaries["frozen_s1_fail"],
        summaries["operational_s1_pass"],
        n_true_rescues=len(rescue_true),
        n_false_rescues=len(rescue_false),
    )

    payload = {
        "status": "retrospective prior-gate softening promotion audit",
        "owner": OWNER,
        "protocol": "protocols/PPX_PRIOR_GATE_SOFTENING_A_PROTOCOL.md",
        "archive": str(SOURCE.relative_to(ROOT)),
        "note": (
            "Stage-1 failure is counterfactual (forced). Archive lacks "
            "prior-only predictions, so OOF prior regret is not recomputed. "
            "Policy A is scored only through stage-2 on prior+residual vs direct."
        ),
        "thresholds": paper_ppx_approval_thresholds(),
        "summaries": summaries,
        "promotion": verdict,
        "rescue_under_forced_s1_fail": {
            "n_rescued_accepts": len(rescue_rows),
            "true_rescues": [row["dataset"] for row in rescue_true],
            "false_rescues": [row["dataset"] for row in rescue_false],
            "n_true_rescues": len(rescue_true),
            "n_false_rescues": len(rescue_false),
        },
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"out": str(OUT / "results.json"), "promotion": verdict}, indent=2))


if __name__ == "__main__":
    main()
