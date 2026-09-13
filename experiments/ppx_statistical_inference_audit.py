#!/usr/bin/env python3
"""Small-n nonparametric statistical audit for PP-X reporting.

Fills gaps relative to journal_statistical_evidence_v1:
- Wilcoxon signed-rank as auxiliary paired test
- explicit n_unit inference tiers (L0/L1/L2)
- dataset-bootstrap CIs for frozen unit-risk policy metrics

Inference units are physical units and datasets — never rows.
See protocols/PPX_STATISTICAL_INFERENCE_PROTOCOL.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

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
OUT = ROOT / "results/ppx_statistical_inference_audit_v1"
OWNER = "박진서"
RNG = np.random.default_rng(20260913)
BOOT = 20_000


def unit_tier(n_units: int) -> str:
    if n_units <= 2:
        return "L0_descriptive"
    if n_units <= 20:
        return "L1_exact"
    return "L2_bootstrap"


def exact_signflip(delta: np.ndarray) -> float:
    delta = np.asarray(delta, float)
    n = len(delta)
    if n == 0:
        return 1.0
    observed = abs(float(delta.mean()))
    if n <= 20:
        masks = np.arange(1 << n, dtype=np.uint64)[:, None]
        signs = 2 * (((masks >> np.arange(n, dtype=np.uint64)) & 1).astype(float)) - 1
        return float(np.mean(np.abs(signs @ delta / n) >= observed - 1e-15))
    signs = RNG.choice((-1.0, 1.0), size=(BOOT, n))
    return float((1 + np.sum(np.abs(signs @ delta / n) >= observed)) / (BOOT + 1))


def wilcoxon_p(delta: np.ndarray) -> float | None:
    delta = np.asarray(delta, float)
    # Drop exact zeros; Wilcoxon needs variation.
    nonzero = delta[np.abs(delta) > 1e-15]
    if len(nonzero) < 3:
        return None
    try:
        return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def bh_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def unit_log_rmse_delta(y, groups, plain, pp) -> np.ndarray:
    deltas = []
    for unit in np.unique(groups):
        mask = groups == unit
        a = np.sqrt(np.mean((y[mask] - plain[mask]) ** 2))
        b = np.sqrt(np.mean((y[mask] - pp[mask]) ** 2))
        deltas.append(np.log(max(a, 1e-12)) - np.log(max(b, 1e-12)))
    return np.asarray(deltas, float)


def stage2_accepts(validation: dict) -> bool:
    thr = paper_ppx_approval_thresholds()
    contract = PPXContract(
        True, True, False, False, False, "direct_fallback"
    )
    prior = PriorEvidence(True, 0, 0)
    decision = select_paper_ppx(
        contract,
        prior,
        (
            PPXCandidateEvidence("direct_fallback", validation["direct_loss"], 0.0, 1.0),
            PPXCandidateEvidence(
                "unbounded",
                validation["pp_loss"],
                validation["unit_win_fraction"],
                validation["worst_unit_ratio"],
            ),
        ),
        min_relative_improvement=thr["min_relative_improvement"],
        min_unit_win_fraction=thr["min_unit_win_fraction"],
        max_worst_unit_rmse_ratio=thr["max_worst_unit_rmse_ratio"],
    )
    return decision.executor == "unbounded"


def policy_metrics(selected: np.ndarray, helped: np.ndarray, effects: np.ndarray) -> dict:
    selected = np.asarray(selected, bool)
    helped = np.asarray(helped, bool)
    effects = np.asarray(effects, float)
    deployed = np.where(selected, effects, 0.0)
    return {
        "accepted": int(selected.sum()),
        "accuracy": float(np.mean(selected == helped)),
        "false_accepts": int(np.sum(selected & ~helped)),
        "false_rejects": int(np.sum(~selected & helped)),
        "deployed_mean_log_rmse_ratio": float(np.mean(deployed)),
    }


def bootstrap_policy(selected, helped, effects, draws: int = BOOT) -> dict:
    n = len(selected)
    keys = [
        "accuracy",
        "false_accepts",
        "false_rejects",
        "deployed_mean_log_rmse_ratio",
    ]
    store = {k: np.empty(draws) for k in keys}
    for i in range(draws):
        idx = RNG.integers(0, n, n)
        m = policy_metrics(selected[idx], helped[idx], effects[idx])
        for k in keys:
            store[k][i] = m[k]
    return {
        k: {
            "point": float(policy_metrics(selected, helped, effects)[k]),
            "ci95": np.quantile(store[k], [0.025, 0.975]).tolist(),
        }
        for k in keys
    }


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    rows = []
    deltas = []
    selected = []
    helped = []
    effects = []
    for name in metadata["datasets"]:
        saved = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        y = np.asarray(saved["y"], float)
        groups = np.asarray(saved["groups"])
        plain = np.asarray(saved["plain"], float).mean(0)
        pp = np.asarray(saved["pp"], float).mean(0)
        delta = unit_log_rmse_delta(y, groups, plain, pp)
        n_u = int(len(delta))
        tier = unit_tier(n_u)
        sign_p = exact_signflip(delta)
        wil_p = wilcoxon_p(delta)
        ci = np.quantile(
            RNG.choice(delta, size=(BOOT, max(n_u, 1)), replace=True).mean(1)
            if n_u
            else np.zeros(BOOT),
            [0.025, 0.975],
        ).tolist()
        validation = validation_evidence(saved)
        test = test_effect(saved)
        accept = stage2_accepts(validation)
        row = {
            "dataset": name,
            "n_units": n_u,
            "inference_tier": tier,
            "unit_wins": int(np.sum(delta > 0)),
            "mean_log_rmse_ratio": float(delta.mean()) if n_u else 0.0,
            "median_log_rmse_ratio": float(np.median(delta)) if n_u else 0.0,
            "unit_bootstrap_ci95": ci,
            "paired_signflip_p": sign_p,
            "wilcoxon_signed_rank_p": wil_p,
            "confirmatory_p_allowed": tier != "L0_descriptive",
            "stage2_accepts_pp": accept,
            "test_pp_helped": bool(test["pp_helped"]),
            "test_mean_unit_log_rmse_ratio": float(test["mean_unit_log_rmse_ratio"]),
            "sign_wilcoxon_agree": (
                None
                if wil_p is None
                else bool((sign_p < 0.05) == (wil_p < 0.05))
            ),
        }
        rows.append(row)
        deltas.append(delta)
        selected.append(accept)
        helped.append(bool(test["pp_helped"]))
        effects.append(float(test["mean_unit_log_rmse_ratio"]))

    # BH only on domains where confirmatory p is allowed.
    conf_idx = [i for i, r in enumerate(rows) if r["confirmatory_p_allowed"]]
    adj = bh_adjust([rows[i]["paired_signflip_p"] for i in conf_idx])
    for i, q in zip(conf_idx, adj):
        rows[i]["bh_fdr_adjusted_signflip_p"] = q
    for i, r in enumerate(rows):
        if "bh_fdr_adjusted_signflip_p" not in r:
            r["bh_fdr_adjusted_signflip_p"] = None

    selected_arr = np.asarray(selected, bool)
    helped_arr = np.asarray(helped, bool)
    effects_arr = np.asarray(effects, float)
    domain_means = np.asarray([r["mean_log_rmse_ratio"] for r in rows])
    domain_wins = int(np.sum(domain_means > 0))

    # Hierarchical bootstrap of equal-dataset mean.
    hier = np.empty(BOOT)
    for b in range(BOOT):
        pick = RNG.integers(0, len(deltas), len(deltas))
        hier[b] = float(
            np.mean(
                [
                    float(np.mean(RNG.choice(deltas[j], len(deltas[j]), replace=True)))
                    if len(deltas[j])
                    else 0.0
                    for j in pick
                ]
            )
        )

    payload = {
        "status": "retrospective nonparametric inference audit",
        "owner": OWNER,
        "protocol": "protocols/PPX_STATISTICAL_INFERENCE_PROTOCOL.md",
        "archive": str(SOURCE.relative_to(ROOT)),
        "effect_definition": (
            "unit log-RMSE ratio = log(direct) - log(PP); positive means PP better"
        ),
        "nonparametric_rule": (
            "Small n_u => exact/nonparametric inference; do not switch the "
            "predictor class solely because n is small."
        ),
        "domain_summary": {
            "n_datasets": len(rows),
            "domain_tier": "D1" if len(rows) >= 5 else "D0",
            "domain_wins": domain_wins,
            "exact_binomial_sign_p": float(
                binomtest(domain_wins, len(rows), 0.5).pvalue
            ),
            "equal_dataset_mean": float(domain_means.mean()),
            "hierarchical_bootstrap_ci95": np.quantile(hier, [0.025, 0.975]).tolist(),
        },
        "tier_counts": {
            tier: int(sum(r["inference_tier"] == tier for r in rows))
            for tier in ("L0_descriptive", "L1_exact", "L2_bootstrap")
        },
        "wilcoxon_signflip_agreement": {
            "n_comparable": int(sum(r["sign_wilcoxon_agree"] is not None for r in rows)),
            "n_agree": int(sum(r["sign_wilcoxon_agree"] is True for r in rows)),
            "n_disagree": int(sum(r["sign_wilcoxon_agree"] is False for r in rows)),
        },
        "operational_policy_bootstrap": {
            "point": policy_metrics(selected_arr, helped_arr, effects_arr),
            "dataset_bootstrap_ci95": bootstrap_policy(
                selected_arr, helped_arr, effects_arr
            ),
        },
        "gaps_remaining": [
            "Archive lacks prior-only predictions; stage-1 OOF prior regret cannot be nonparametric-tested here.",
            "L0 domains must stay descriptive even if point estimates look large.",
            "This audit is retrospective common-backbone, not prospective confirmation.",
        ],
        "datasets": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "out": str(OUT / "results.json"),
                "domain_summary": payload["domain_summary"],
                "tier_counts": payload["tier_counts"],
                "agreement": payload["wilcoxon_signflip_agreement"],
                "policy": payload["operational_policy_bootstrap"]["point"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
