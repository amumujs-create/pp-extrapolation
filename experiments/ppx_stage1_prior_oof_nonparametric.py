#!/usr/bin/env python3
"""Stage-1 prior OOF regret vs matched direct — nonparametric audit.

Fits train-LOO ridge affine prior (no validation peek for alpha), scores it on
the same validation units as the frozen common-backbone plain ensemble, and
applies exact/nonparametric tests from PPX_STATISTICAL_INFERENCE_PROTOCOL.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from pp_extrapolation.model import (  # noqa: E402
    _affine_prediction,
    _arrays,
    equal_group_weights,
    fit_feature_scale,
    solve_weighted_affine,
    transform_features,
)
from pp_extrapolation.paper_ppx import (  # noqa: E402
    PPXCandidateEvidence,
    PPXContract,
    select_paper_ppx,
)
from pp_extrapolation.presets import paper_ppx_approval_thresholds  # noqa: E402
from pp_extrapolation.transferability_gate import PriorEvidence  # noqa: E402
from ppx_paper_policy_audit import validation_evidence  # noqa: E402

SOURCE = ROOT / "results/cross_domain_mechanism_v2"
OUT = ROOT / "results/ppx_stage1_prior_oof_nonparametric_v1"
OWNER = "박진서"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
RNG = np.random.default_rng(20260913)
BOOT = 20_000


def unit_tier(n: int) -> str:
    if n <= 2:
        return "L0_descriptive"
    if n <= 20:
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
    nonzero = np.asarray(delta, float)
    nonzero = nonzero[np.abs(nonzero) > 1e-15]
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
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out.tolist()


def select_alpha_train_loo(train: dict) -> dict:
    """Choose ridge alpha by train-unit LOO MSE (no validation labels)."""
    x, y, groups = _arrays(train)
    units = np.unique(groups)
    center, scale = fit_feature_scale(x)
    z = transform_features(x, center, scale)
    target_scale = max(float(np.max(y)), 1.0)
    y_scaled = y / target_scale
    scores = []
    for alpha in ALPHAS:
        oof = np.empty(len(y), float)
        for unit in units:
            hold = groups == unit
            fit_mask = ~hold
            if not fit_mask.any():
                continue
            weights = equal_group_weights(groups[fit_mask])
            sol = solve_weighted_affine(
                z[fit_mask], y_scaled[fit_mask], weights, alpha=float(alpha)
            )
            oof[hold] = _affine_prediction(
                sol, x[hold], center, scale, target_scale
            )
        # Equal-unit mean MSE (not row-pooled).
        unit_mse = [
            float(np.mean((y[groups == u] - oof[groups == u]) ** 2)) for u in units
        ]
        scores.append({"alpha": float(alpha), "unit_macro_mse": float(np.mean(unit_mse))})
    best = min(scores, key=lambda r: (r["unit_macro_mse"], r["alpha"]))
    weights = equal_group_weights(groups)
    solution = solve_weighted_affine(
        z, y_scaled, weights, alpha=best["alpha"]
    )
    return {
        "center": center,
        "scale": scale,
        "target_scale": target_scale,
        "initialization": solution,
        "selected_alpha": best["alpha"],
        "loo_candidates": scores,
        "n_train_units": int(len(units)),
    }


def predict_affine(selection: dict, x: np.ndarray) -> np.ndarray:
    return _affine_prediction(
        selection["initialization"],
        x,
        selection["center"],
        selection["scale"],
        selection["target_scale"],
    )


def unit_rmse(y, groups, pred) -> np.ndarray:
    return np.asarray(
        [
            np.sqrt(np.mean((y[groups == u] - pred[groups == u]) ** 2))
            for u in np.unique(groups)
        ],
        float,
    )


def stage2_accepts(archive) -> bool:
    validation = validation_evidence(archive)
    thr = paper_ppx_approval_thresholds()
    decision = select_paper_ppx(
        PPXContract(True, True, False, False, False, "direct_fallback"),
        PriorEvidence(True, 0, 0),
        (
            PPXCandidateEvidence(
                "direct_fallback", validation["direct_loss"], 0.0, 1.0
            ),
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


def load_splits() -> dict[str, tuple[dict, dict]]:
    from extrapolation_competitors_all import datasets as generic_datasets

    raw = generic_datasets()
    out = {k: (v[0], v[1]) for k, v in raw.items()}

    from run_affine_tail_external_nasa_health_v2 import prepare_folds

    folds = prepare_folds()[0]
    # Concatenate LOO folds' train/val like the archive aggregation for scoring
    # prior on each fold's validation, then stack — matches nasa archive layout.
    tr_parts, va_parts = [], []
    for fold in folds:
        tr_parts.append(fold["train"])
        va_parts.append(fold["validation"])
    # For alpha: use first fold train only would bias; fit per-fold below instead.
    out["nasa_battery"] = ("__nasa_folds__", folds)

    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_tra_quantile_split import make_tra_hard_split
    from ncmapss_pp_benchmark import rows

    split = make_tra_hard_split(
        (ROOT / "data/N-CMAPSS_DS02-006.h5").resolve(),
        max_windows_per_unit=1500,
        random_seed=42,
    )
    names = list(FEATURE_COLS)
    out["ncmapss"] = (rows(split.train, names), rows(split.val, names))
    return out


def score_pair(name: str, train: dict, validation: dict, archive) -> dict:
    t0 = time.time()
    selection = select_alpha_train_loo(train)
    prior = predict_affine(selection, np.asarray(validation["x"], float))
    y = np.asarray(validation["y"], float)
    groups = np.asarray(archive["validation_groups"])
    # Archive row order must match validation rows used in mechanism study.
    if len(groups) != len(y):
        groups = np.asarray(validation["groups"])
    direct = np.asarray(archive["validation_plain"], float).mean(0)
    if len(direct) != len(y):
        raise ValueError(f"{name}: archive validation length mismatch")

    prior_mse = float(np.mean((y - prior) ** 2))
    direct_mse = float(np.mean((y - direct) ** 2))
    regret = prior_mse - direct_mse
    prior_u = unit_rmse(y, groups, prior)
    direct_u = unit_rmse(y, groups, direct)
    # Positive delta => prior better (lower RMSE).
    delta = np.log(np.maximum(direct_u, 1e-12)) - np.log(np.maximum(prior_u, 1e-12))
    n_u = len(delta)
    tier = unit_tier(n_u)
    sign_p = exact_signflip(delta)
    wil_p = wilcoxon_p(delta)
    ci = np.quantile(
        RNG.choice(delta, size=(BOOT, n_u), replace=True).mean(1), [0.025, 0.975]
    ).tolist()
    stage1_pass = regret <= 0.0
    s2 = stage2_accepts(archive)
    test_y = np.asarray(archive["y"], float)
    test_g = np.asarray(archive["groups"])
    test_plain = np.asarray(archive["plain"], float).mean(0)
    test_pp = np.asarray(archive["pp"], float).mean(0)
    test_plain_u = unit_rmse(test_y, test_g, test_plain)
    test_pp_u = unit_rmse(test_y, test_g, test_pp)
    test_helped = bool(np.mean(test_plain_u - test_pp_u) > 0)

    return {
        "dataset": name,
        "seconds": round(time.time() - t0, 2),
        "n_train_units": selection["n_train_units"],
        "n_validation_units": n_u,
        "inference_tier": tier,
        "selected_alpha": selection["selected_alpha"],
        "prior_mse": prior_mse,
        "direct_mse": direct_mse,
        "oof_prior_regret": regret,
        "stage1_pass_regret_nonpositive": stage1_pass,
        "unit_wins_prior_vs_direct": int(np.sum(delta > 0)),
        "mean_log_rmse_ratio_prior_vs_direct": float(delta.mean()),
        "unit_bootstrap_ci95": ci,
        "paired_signflip_p": sign_p,
        "wilcoxon_signed_rank_p": wil_p,
        "confirmatory_p_allowed": tier != "L0_descriptive",
        "stage2_accepts_pp": s2,
        "test_pp_helped": test_helped,
        "loo_candidates": selection["loo_candidates"],
    }


def score_nasa(folds, archive) -> dict:
    """Per-fold train-LOO affine on that fold's validation, then stack."""
    t0 = time.time()
    priors, directs, ys, groups = [], [], [], []
    alphas = []
    for fold in folds:
        sel = select_alpha_train_loo(fold["train"])
        alphas.append(sel["selected_alpha"])
        va = fold["validation"]
        priors.append(predict_affine(sel, np.asarray(va["x"], float)))
        ys.append(np.asarray(va["y"], float))
        groups.append(np.asarray(va["groups"]))
    # Archive validation_plain is concatenated in fold order (mechanism study).
    prior = np.concatenate(priors)
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    direct = np.asarray(archive["validation_plain"], float).mean(0)
    if len(direct) != len(y):
        raise ValueError("nasa_battery validation length mismatch")

    # Build a fake archive-like dict for stage2 using full archive.
    regret_row = {
        "validation_y": y,
        "validation_groups": g,
        "validation_plain": np.asarray(archive["validation_plain"], float),
        "validation_pp": np.asarray(archive["validation_pp"], float),
        "y": archive["y"],
        "groups": archive["groups"],
        "plain": archive["plain"],
        "pp": archive["pp"],
    }
    # Reuse scoring math
    prior_mse = float(np.mean((y - prior) ** 2))
    direct_mse = float(np.mean((y - direct) ** 2))
    regret = prior_mse - direct_mse
    prior_u = unit_rmse(y, g, prior)
    direct_u = unit_rmse(y, g, direct)
    delta = np.log(np.maximum(direct_u, 1e-12)) - np.log(np.maximum(prior_u, 1e-12))
    n_u = len(delta)
    tier = unit_tier(n_u)
    test_y = np.asarray(archive["y"], float)
    test_g = np.asarray(archive["groups"])
    test_plain_u = unit_rmse(test_y, test_g, np.asarray(archive["plain"], float).mean(0))
    test_pp_u = unit_rmse(test_y, test_g, np.asarray(archive["pp"], float).mean(0))
    return {
        "dataset": "nasa_battery",
        "seconds": round(time.time() - t0, 2),
        "n_train_units": "fold-specific",
        "n_validation_units": n_u,
        "inference_tier": tier,
        "selected_alpha": float(np.median(alphas)),
        "fold_alphas": alphas,
        "prior_mse": prior_mse,
        "direct_mse": direct_mse,
        "oof_prior_regret": regret,
        "stage1_pass_regret_nonpositive": regret <= 0.0,
        "unit_wins_prior_vs_direct": int(np.sum(delta > 0)),
        "mean_log_rmse_ratio_prior_vs_direct": float(delta.mean()),
        "unit_bootstrap_ci95": np.quantile(
            RNG.choice(delta, size=(BOOT, n_u), replace=True).mean(1), [0.025, 0.975]
        ).tolist(),
        "paired_signflip_p": exact_signflip(delta),
        "wilcoxon_signed_rank_p": wilcoxon_p(delta),
        "confirmatory_p_allowed": tier != "L0_descriptive",
        "stage2_accepts_pp": stage2_accepts(regret_row),
        "test_pp_helped": bool(np.mean(test_plain_u - test_pp_u) > 0),
    }


def main() -> None:
    metadata = json.loads((SOURCE / "results.json").read_text())
    splits = load_splits()
    rows = []
    for name in metadata["datasets"]:
        print(f"=== {name} ===", flush=True)
        archive = np.load(SOURCE / f"{name}_predictions.npz", allow_pickle=True)
        item = splits[name]
        if name == "nasa_battery":
            row = score_nasa(item[1], archive)
        else:
            train, validation = item
            # Align groups from split if archive groups differ in dtype only.
            row = score_pair(name, train, validation, archive)
        print(
            name,
            "regret",
            round(row["oof_prior_regret"], 4),
            "pass",
            row["stage1_pass_regret_nonpositive"],
            "sign_p",
            round(row["paired_signflip_p"], 4),
            flush=True,
        )
        rows.append(row)

    conf = [r for r in rows if r["confirmatory_p_allowed"]]
    adj = bh_adjust([r["paired_signflip_p"] for r in conf])
    for r, q in zip(conf, adj):
        r["bh_fdr_adjusted_signflip_p"] = q
    for r in rows:
        r.setdefault("bh_fdr_adjusted_signflip_p", None)

    passes = [r for r in rows if r["stage1_pass_regret_nonpositive"]]
    fails = [r for r in rows if not r["stage1_pass_regret_nonpositive"]]
    # Concordance tables
    s1 = np.asarray([r["stage1_pass_regret_nonpositive"] for r in rows])
    s2 = np.asarray([r["stage2_accepts_pp"] for r in rows])
    helped = np.asarray([r["test_pp_helped"] for r in rows])
    prior_better_mean = np.asarray(
        [r["mean_log_rmse_ratio_prior_vs_direct"] > 0 for r in rows]
    )

    payload = {
        "status": "retrospective stage-1 prior OOF nonparametric audit",
        "owner": OWNER,
        "protocol": "protocols/PPX_STAGE1_PRIOR_OOF_NONPARAMETRIC_PROTOCOL.md",
        "inference_protocol": "protocols/PPX_STATISTICAL_INFERENCE_PROTOCOL.md",
        "comparison": {
            "prior": "train-unit LOO ridge affine (alpha without validation labels)",
            "direct": "archive validation_plain ensemble",
        },
        "summary": {
            "n_datasets": len(rows),
            "stage1_pass": int(s1.sum()),
            "stage1_fail": int((~s1).sum()),
            "domains_prior_mean_better": int(prior_better_mean.sum()),
            "exact_binomial_prior_mean_better_p": float(
                binomtest(int(prior_better_mean.sum()), len(rows), 0.5).pvalue
            ),
            "stage1_and_stage2_both_accept": int(np.sum(s1 & s2)),
            "stage1_pass_stage2_reject": int(np.sum(s1 & ~s2)),
            "stage1_fail_stage2_accept": int(np.sum(~s1 & s2)),
            "stage1_pass_and_test_helped": int(np.sum(s1 & helped)),
            "stage1_fail_but_test_helped": int(np.sum(~s1 & helped)),
        },
        "pass_datasets": [r["dataset"] for r in passes],
        "fail_datasets": [r["dataset"] for r in fails],
        "datasets": rows,
        "guardrails": [
            "Does not retune frozen Algorithm-1 thresholds.",
            "Does not promote Policy A; only reconstructs stage-1 evidence.",
            "Validation scoring uses group-disjoint units; alpha uses train LOO only.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"out": str(OUT / "results.json"), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
