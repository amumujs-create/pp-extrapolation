#!/usr/bin/env python3
"""Nested source-only GCIE v5 development experiment on N-CMAPSS DS03.

The DS03 test outcomes have already been revealed by earlier retrospective
development.  This run is therefore not prospective, but it preserves a
test-label-blind execution contract: nested train-only evidence freezes the
configuration and distribution gate before test is loaded and scored once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ncmapss_ds03_gcie_v4 import (
    CONFIGS,
    ENGRESSION,
    _direct_matrix,
    _engression_metrics,
    _metrics,
    _repeat_direct,
    _split_audit,
    _write_json,
    subset,
)
from pp_extrapolation.ds03_prospective import (
    SEEDS,
    TEST_UNITS,
    TRAIN_UNITS,
    VALIDATION_UNITS,
    _fit_direct,
    _predict_direct,
    causal_features,
    load_development,
    load_revealed_test,
)
from pp_extrapolation.grade_cvar_implicit_expert import (
    GCIEConfig,
    fit_gcie,
    predict_samples,
)
from pp_extrapolation.pseudo_extrapolation import (
    build_nested_pseudo_extrapolation_episodes,
    fit_support_grade as fit_source_support_grade,
    transform_support_grade,
)
from pp_extrapolation.source_only_distribution_gate import (
    SourceOnlyDistributionGate,
    apply_source_only_distribution_gate,
    select_source_only_distribution_gate,
)


SEARCH_SEED = 42
CUTOFFS = (0.5, 0.65, 0.8)
MINIMUM_UNITS = 6
MINIMUM_MACRO_IMPROVEMENT = 0.02
MINIMUM_WIN_FRACTION = 0.60
MAXIMUM_WORST_REGRET = 0.02
MAXIMUM_CVAR20_REGRET = 0.0
DEFAULT_OUT = ROOT / "results/ncmapss_ds03_nested_gcie_v5_dev"
DEFAULT_SELECTION = ROOT / "results/ncmapss_ds03_ppx_v1/selection.json"


def _take(rows: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in rows.items()}


def deterministic_inner_unit(
    outer_unit: str,
    available: set[tuple[str, str, float]] | None = None,
) -> str:
    """Choose the first cyclic inner unit with all requested pseudo-tails."""
    units = tuple(str(unit) for unit in TRAIN_UNITS)
    index = units.index(str(outer_unit))
    candidates = (
        units[(index + offset) % len(units)]
        for offset in range(1, len(units))
    )
    for inner in candidates:
        if available is None or all(
            (str(outer_unit), inner, cutoff) in available
            for cutoff in CUTOFFS
        ):
            return inner
    raise ValueError(
        f"no inner unit gives non-empty evidence at every cutoff for outer {outer_unit}"
    )


def _fit_gcie_rows(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    config: GCIEConfig,
):
    return fit_gcie(
        train["x"],
        train["y"],
        train["groups"],
        train["cycles"],
        validation["x"],
        validation["y"],
        validation["groups"],
        validation["cycles"],
        config=config,
    )


def _candidate_metric(
    fit: Any, validation: dict[str, np.ndarray], *, seed: int
) -> dict[str, float]:
    samples = predict_samples(
        fit,
        validation["x"],
        sample_count=fit.config.sample_count,
        seed=seed,
    )
    prediction = samples.mean(axis=1)
    return {
        "rmse": math.sqrt(float(np.mean((validation["y"] - prediction) ** 2))),
        "energy_score": float(
            np.mean(np.abs(samples - validation["y"][:, None]))
            - 0.5
            * np.mean(
                np.abs(samples[:, :, None] - samples[:, None, :])
            )
        ),
    }


def _config_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row["cutoff_averaged_rmse"]),
        float(row["cutoff_averaged_energy_score"]),
        int(row["candidate"]),
    )


def nested_config_evidence(
    train: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], Any]:
    """Select one config per outer split using disjoint inner pseudo-tails."""
    plan = build_nested_pseudo_extrapolation_episodes(
        train["groups"],
        train["cycles"],
        outer_rule="high",
        inner_rule="high",
        cutoffs=CUTOFFS,
    )
    lookup = {
        (str(episode.outer_unit), str(episode.inner_unit), float(episode.cutoff)): episode
        for episode in plan.episodes
    }
    outer_rows: list[dict[str, Any]] = []
    for outer in (str(unit) for unit in TRAIN_UNITS):
        inner = deterministic_inner_unit(outer, set(lookup))
        candidates = []
        for candidate_index, config in enumerate(CONFIGS):
            cutoff_rows = []
            for cutoff_index, cutoff in enumerate(CUTOFFS):
                episode = lookup[(outer, inner, cutoff)]
                fit_rows = _take(train, episode.fit_indices)
                tune_rows = _take(train, episode.tune_indices)
                fit = _fit_gcie_rows(fit_rows, tune_rows, config)
                metric = _candidate_metric(
                    fit,
                    tune_rows,
                    seed=SEARCH_SEED + 10_000 + cutoff_index,
                )
                cutoff_rows.append(
                    {
                        "cutoff": cutoff,
                        "threshold": episode.threshold,
                        "fit_rows": int(len(episode.fit_indices)),
                        "tune_rows": int(len(episode.tune_indices)),
                        "fit_units": [str(unit) for unit in episode.fit_units],
                        "selected_epoch": fit.selection["selected_epoch"],
                        "epochs_executed": fit.selection["epochs_executed"],
                        **metric,
                    }
                )
            candidates.append(
                {
                    "candidate": candidate_index,
                    "config": asdict(config),
                    "cutoff_averaged_rmse": float(
                        np.mean([row["rmse"] for row in cutoff_rows])
                    ),
                    "cutoff_averaged_energy_score": float(
                        np.mean([row["energy_score"] for row in cutoff_rows])
                    ),
                    "cutoffs": cutoff_rows,
                    "physical_unit_vote_count": 1,
                    "repeated_cutoffs_counted_as_independent_units": False,
                }
            )
        selected = min(candidates, key=_config_key)
        outer_rows.append(
            {
                "outer_unit": outer,
                "inner_unit": inner,
                "selected_candidate": int(selected["candidate"]),
                "selection_rule": (
                    "minimum cutoff-averaged inner-unit RMSE; then energy; "
                    "then candidate index"
                ),
                "outer_label_used": False,
                "candidates": candidates,
            }
        )
        print(
            "NESTED",
            f"outer={outer}",
            f"inner={inner}",
            f"config={selected['candidate']}",
            f"rmse={selected['cutoff_averaged_rmse']:.6f}",
            flush=True,
        )
    return outer_rows, plan


def select_global_config(
    outer_evidence: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    """Aggregate six outer-specific inner risks with one vote per outer unit."""
    aggregate = []
    for candidate_index in range(len(CONFIGS)):
        risks = [
            row["candidates"][candidate_index]["cutoff_averaged_rmse"]
            for row in outer_evidence
        ]
        energies = [
            row["candidates"][candidate_index]["cutoff_averaged_energy_score"]
            for row in outer_evidence
        ]
        aggregate.append(
            {
                "candidate": candidate_index,
                "mean_outer_risk": float(np.mean(risks)),
                "worst_outer_risk": float(np.max(risks)),
                "mean_outer_energy_score": float(np.mean(energies)),
                "outer_unit_count": len(risks),
            }
        )
    selected = min(
        aggregate,
        key=lambda row: (
            row["mean_outer_risk"],
            row["worst_outer_risk"],
            row["mean_outer_energy_score"],
            row["candidate"],
        ),
    )
    return int(selected["candidate"]), aggregate


def _outer_training_split(
    train: dict[str, np.ndarray], outer: str, inner: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit_units = tuple(
        int(unit)
        for unit in TRAIN_UNITS
        if str(unit) not in {str(outer), str(inner)}
    )
    return (
        subset(train, fit_units),
        subset(train, (int(inner),)),
        subset(train, (int(outer),)),
    )


def build_outer_oof(
    train: dict[str, np.ndarray],
    outer_evidence: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Generate row-aligned outer-unit OOF direct and GCIE distributions."""
    sample_count = CONFIGS[0].sample_count * len(SEEDS)
    fallback_samples = np.empty((len(train["y"]), sample_count), dtype=np.float64)
    candidate_samples = np.empty_like(fallback_samples)
    source_grades = np.empty(len(train["y"]), dtype=np.float64)
    filled = np.zeros(len(train["y"]), dtype=bool)
    audit = []
    for row in outer_evidence:
        outer = str(row["outer_unit"])
        inner = str(row["inner_unit"])
        candidate_index = int(row["selected_candidate"])
        config = CONFIGS[candidate_index]
        fit_rows, tune_rows, held_rows = _outer_training_split(
            train, outer, inner
        )
        held_indices = np.flatnonzero(train["groups"] == outer)
        direct_parts = []
        candidate_parts = []
        seed_audit = []
        for seed in SEEDS:
            direct_fit = _fit_direct(fit_rows, tune_rows, seed)
            direct_prediction = _predict_direct(direct_fit, held_rows["x"])
            direct_parts.append(
                np.repeat(
                    direct_prediction[:, None], config.sample_count, axis=1
                )
            )
            gcie_fit = _fit_gcie_rows(
                fit_rows, tune_rows, replace(config, seed=seed)
            )
            candidate_parts.append(
                predict_samples(
                    gcie_fit,
                    held_rows["x"],
                    sample_count=config.sample_count,
                    seed=seed + 30_000,
                )
            )
            seed_audit.append(
                {
                    "seed": seed,
                    "gcie_selected_epoch": gcie_fit.selection[
                        "selected_epoch"
                    ],
                    "fit_units": sorted(set(fit_rows["groups"].tolist())),
                    "early_stopping_units": sorted(
                        set(tune_rows["groups"].tolist())
                    ),
                }
            )
        grade_fit = fit_source_support_grade(
            fit_rows["x"], fit_rows["groups"], fit_rows["cycles"]
        )
        grades = transform_support_grade(
            grade_fit, held_rows["x"], held_rows["cycles"]
        )
        fallback_samples[held_indices] = np.concatenate(direct_parts, axis=1)
        candidate_samples[held_indices] = np.concatenate(
            candidate_parts, axis=1
        )
        source_grades[held_indices] = grades
        filled[held_indices] = True
        audit.append(
            {
                "outer_unit": outer,
                "inner_unit": inner,
                "candidate": candidate_index,
                "outer_rows": int(len(held_indices)),
                "fit_units": sorted(set(fit_rows["groups"].tolist())),
                "early_stopping_units": sorted(
                    set(tune_rows["groups"].tolist())
                ),
                "outer_absent_from_fit": outer
                not in set(fit_rows["groups"].tolist()),
                "outer_absent_from_early_stopping": outer
                not in set(tune_rows["groups"].tolist()),
                "outer_absent_from_config_evidence": bool(
                    row["outer_label_used"] is False
                ),
                "support_source_groups": [
                    str(group) for group in grade_fit.source_groups
                ],
                "outer_absent_from_support_fit": outer
                not in set(map(str, grade_fit.source_groups)),
                "grade_minimum": float(np.min(grades)),
                "grade_maximum": float(np.max(grades)),
                "seeds": seed_audit,
            }
        )
    if not np.all(filled):
        raise RuntimeError("outer OOF construction left unfilled source rows")
    return (
        {
            "fallback_samples": fallback_samples,
            "candidate_samples": candidate_samples,
            "fallback_mean": fallback_samples.mean(axis=1),
            "candidate_mean": candidate_samples.mean(axis=1),
            "source_grades": source_grades,
        },
        audit,
    )


def gated_samples(
    gate: SourceOnlyDistributionGate,
    fallback_samples: np.ndarray,
    candidate_samples: np.ndarray,
    grades: np.ndarray,
) -> np.ndarray:
    """Apply the point gate row-wise to aligned samples, preserving fallback."""
    fallback = np.asarray(fallback_samples)
    candidate = np.asarray(candidate_samples, dtype=np.float64)
    grades = np.asarray(grades, dtype=np.float64)
    if fallback.shape != candidate.shape or fallback.ndim != 2:
        raise ValueError("fallback and candidate samples must align")
    if grades.shape != (len(fallback),):
        raise ValueError("grades must align with sample rows")
    if gate.alpha == 0.0:
        return fallback.copy()
    result = fallback.astype(np.float64, copy=True)
    inside = grades <= gate.source_grade_max
    result[inside] += gate.alpha * (
        candidate[inside] - result[inside]
    )
    return result


def _final_candidate_samples(
    fits: list[Any], rows: dict[str, np.ndarray]
) -> np.ndarray:
    return np.concatenate(
        [
            predict_samples(
                fit,
                rows["x"],
                sample_count=fit.config.sample_count,
                seed=int(fit.config.seed) + 30_000,
            )
            for fit in fits
        ],
        axis=1,
    )


def _gate_dict(gate: SourceOnlyDistributionGate) -> dict[str, Any]:
    return asdict(gate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", type=Path, default=ROOT / "data/N-CMAPSS_DS03-012.h5"
    )
    parser.add_argument(
        "--selection", type=Path, default=DEFAULT_SELECTION
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prior_failed_freeze = None
    if args.out.exists():
        result_path = args.out / "results.json"
        failed_freeze_path = args.out / "selection.json"
        if result_path.exists() or not failed_freeze_path.exists():
            raise FileExistsError(
                f"immutable nested GCIE v5 output already exists: {args.out}"
            )
        prior_failed_freeze = json.loads(failed_freeze_path.read_text())
    started = time.monotonic()
    torch.set_num_threads(2)

    prospective = json.loads(args.selection.read_text())
    if prospective["selected_route"] != "direct_fallback" or prospective["approved"]:
        raise ValueError("nested GCIE v5 requires the prior-rejected branch")
    bundle = torch.load(
        args.selection.with_name(prospective["model_artifact"]),
        map_location="cpu",
        weights_only=False,
    )
    frozen_direct_fits = bundle["fits"]["direct_fallback"]

    development = causal_features(load_development(args.h5), "direct")
    train = subset(development, TRAIN_UNITS)
    validation = subset(development, VALIDATION_UNITS)

    nested_started = time.monotonic()
    outer_evidence, plan = nested_config_evidence(train)
    global_candidate, aggregate_risk = select_global_config(outer_evidence)
    global_config = CONFIGS[global_candidate]
    nested_seconds = time.monotonic() - nested_started

    oof_started = time.monotonic()
    outer_oof, outer_audit = build_outer_oof(train, outer_evidence)
    gate = select_source_only_distribution_gate(
        train["groups"],
        train["y"],
        outer_oof["fallback_mean"],
        outer_oof["candidate_mean"],
        source_grades=outer_oof["source_grades"],
        candidate_samples=outer_oof["candidate_samples"],
        fallback_samples=outer_oof["fallback_samples"],
        minimum_units=MINIMUM_UNITS,
        minimum_macro_improvement=MINIMUM_MACRO_IMPROVEMENT,
        minimum_win_fraction=MINIMUM_WIN_FRACTION,
        maximum_worst_relative_regret=MAXIMUM_WORST_REGRET,
        maximum_cvar20_relative_regret=MAXIMUM_CVAR20_REGRET,
    )
    oof_gated_mean = apply_source_only_distribution_gate(
        gate,
        outer_oof["fallback_mean"],
        outer_oof["candidate_mean"],
        outer_oof["source_grades"],
    )
    oof_gated_samples = gated_samples(
        gate,
        outer_oof["fallback_samples"],
        outer_oof["candidate_samples"],
        outer_oof["source_grades"],
    )
    if not np.allclose(
        oof_gated_mean, oof_gated_samples.mean(axis=1), rtol=0.0, atol=1e-12
    ):
        raise AssertionError("point and distribution gate applications disagree")
    oof_seconds = time.monotonic() - oof_started

    # Configuration and gate are now frozen. Validation units 7--9 are used
    # only because fit_gcie requires labeled validation for early stopping.
    final_started = time.monotonic()
    final_fits = []
    final_fit_audit = []
    for seed in SEEDS:
        fit = _fit_gcie_rows(
            train, validation, replace(global_config, seed=seed)
        )
        final_fits.append(fit)
        final_fit_audit.append(
            {
                "seed": seed,
                "selected_epoch": fit.selection["selected_epoch"],
                "epochs_executed": fit.selection["epochs_executed"],
                "best_validation_objective": fit.selection[
                    "best_validation_objective"
                ],
            }
        )
    final_gate_support = fit_source_support_grade(
        train["x"], train["groups"], train["cycles"]
    )
    final_seconds = time.monotonic() - final_started

    recovery_audit = None
    if prior_failed_freeze is not None:
        previous_recovery = prior_failed_freeze.get("recovery_audit") or {}
        previous_test_loads = int(
            previous_recovery.get("test_load_attempts_in_artifact_lineage", 1)
        )
        previous_global = prior_failed_freeze["global_config_selection"][
            "selected_candidate"
        ]
        previous_gate = prior_failed_freeze["source_only_distribution_gate"]
        unchanged = (
            int(previous_global) == global_candidate
            and float(previous_gate["alpha"]) == gate.alpha
            and np.isclose(
                float(previous_gate["source_grade_max"]),
                gate.source_grade_max,
                rtol=0.0,
                atol=1e-12,
            )
        )
        if not unchanged:
            raise RuntimeError(
                "recovery recomputation changed the frozen config or gate"
            )
        recovery_audit = {
            "prior_attempt_failed_after_test_load_before_outcome_scoring": True,
            "failure": (
                "point/sample mean numerical-consistency assertion used "
                "an unnecessarily strict 1e-12 tolerance"
            ),
            "config_gate_recomputed_deterministically_unchanged": True,
            "test_load_attempts_in_artifact_lineage": previous_test_loads + 1,
            "successful_test_outcome_evaluations_before_this_attempt": 0,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    frozen_payload = {
        "schema": "ncmapss-ds03-nested-gcie-v5-source-only-freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "Grade-CVaR Implicit Expert (GCIE)",
        "development_status": (
            "retrospective development: DS03 test labels were publicly "
            "revealed by prior experiments, but are unused for this selection"
        ),
        "nested_plan": {
            "cutoffs": list(CUTOFFS),
            "episode_count": len(plan.episodes),
            "audit_passed": plan.audit_passed,
            "outer_rule": plan.outer_rule,
            "inner_rule": plan.inner_rule,
            "deterministic_inner_rule": (
                "first source unit cyclically with non-empty .5/.65/.8 "
                "coordinate-only pseudo-tails"
            ),
            "repeated_cutoffs_collapsed_within_inner_unit": True,
        },
        "outer_config_evidence": outer_evidence,
        "global_config_selection": {
            "selected_candidate": global_candidate,
            "selected_config": asdict(global_config),
            "rule": (
                "minimum mean outer-specific cutoff-averaged inner risk; "
                "then worst risk, energy, candidate index"
            ),
            "aggregate_risk_table": aggregate_risk,
        },
        "source_only_distribution_gate": _gate_dict(gate),
        "gate_thresholds": {
            "minimum_units": MINIMUM_UNITS,
            "minimum_macro_improvement": MINIMUM_MACRO_IMPROVEMENT,
            "minimum_win_fraction": MINIMUM_WIN_FRACTION,
            "maximum_worst_relative_regret": MAXIMUM_WORST_REGRET,
            "maximum_cvar20_relative_regret": MAXIMUM_CVAR20_REGRET,
            "alpha_grid": list(gate.alpha_grid),
            "observed_source_grade_max": gate.source_grade_max,
        },
        "outer_oof_metrics": {
            "gated": _metrics(
                train["y"],
                train["groups"],
                oof_gated_samples,
                outer_oof["fallback_mean"],
            ),
            "fallback": _metrics(
                train["y"],
                train["groups"],
                outer_oof["fallback_samples"],
                outer_oof["fallback_mean"],
            ),
        },
        "outer_exclusion_audit": outer_audit,
        "final_fit": {
            "seeds": list(SEEDS),
            "runs": final_fit_audit,
            "validation_use": (
                "units 7--9 labels used only for final fit_gcie early stopping; "
                "fit API has no train-only checkpoint mode"
            ),
            "validation_used_for_config": False,
            "validation_used_for_gate_alpha": False,
            "support_source_groups": [
                str(group) for group in final_gate_support.source_groups
            ],
        },
        "split_audit": {
            "train": _split_audit(train),
            "validation": _split_audit(validation),
            "train_units_expected": list(TRAIN_UNITS),
            "validation_units_expected": list(VALIDATION_UNITS),
            "test_units_expected": list(TEST_UNITS),
        },
        "information_contract": {
            "config_frozen_before_test_load": True,
            "gate_frozen_before_test_load": True,
            "engression_used_for_tuning": False,
            "test_labels_used_for_tuning": False,
            "application_grade_uses_frozen_source_only_state": True,
            "application_batch_statistics_used": False,
            "full_unit_rank_used": False,
        },
        "recovery_audit": recovery_audit,
        "timing_seconds": {
            "nested_config_selection": nested_seconds,
            "outer_oof_and_gate": oof_seconds,
            "final_refits": final_seconds,
        },
    }
    freeze_path = args.out / "selection.json"
    _write_json(freeze_path, frozen_payload)
    freeze_sha256 = hashlib.sha256(freeze_path.read_bytes()).hexdigest()

    # First and only test load/evaluation in this run.
    test_started = time.monotonic()
    test = causal_features(load_revealed_test(args.h5), "direct")
    if tuple(map(int, np.unique(test["groups"]))) != TEST_UNITS:
        raise ValueError("unexpected test physical units")
    direct_matrix = _direct_matrix(test, frozen_direct_fits)
    direct_samples = _repeat_direct(
        direct_matrix, global_config.sample_count
    )
    direct_mean = direct_samples.astype(np.float64).mean(axis=1)
    candidate_samples = _final_candidate_samples(final_fits, test)
    candidate_mean = candidate_samples.mean(axis=1)
    test_grades = transform_support_grade(
        final_gate_support, test["x"], test["cycles"]
    )
    gated_mean = apply_source_only_distribution_gate(
        gate, direct_mean, candidate_mean, test_grades
    )
    output_samples = gated_samples(
        gate, direct_samples, candidate_samples, test_grades
    )
    if not np.allclose(
        gated_mean, output_samples.mean(axis=1), rtol=0.0, atol=1e-9
    ):
        raise AssertionError("test point and distribution gates disagree")
    outside = test_grades > gate.source_grade_max
    if not np.array_equal(output_samples[outside], direct_samples[outside]):
        raise AssertionError("out-of-grade rows lost exact fallback identity")

    gcie_metrics = _metrics(
        test["y"], test["groups"], output_samples, direct_mean
    )
    direct_metrics = _metrics(
        test["y"], test["groups"], direct_samples, direct_mean
    )
    engression_metrics, engression_audit = _engression_metrics(
        test, direct_mean
    )
    beats_engression = (
        gcie_metrics["pooled_r2"] > engression_metrics["pooled_r2"]
        and gcie_metrics["macro_rmse"] < engression_metrics["macro_rmse"]
    )
    regret_safe = (
        gcie_metrics["worst_unit_regret_relative_rmse_vs_direct"]
        <= MAXIMUM_WORST_REGRET + 1e-12
    )
    accepted = beats_engression and regret_safe
    failure_reasons = []
    if gcie_metrics["pooled_r2"] <= engression_metrics["pooled_r2"]:
        failure_reasons.append("pooled R2 did not exceed Engression")
    if gcie_metrics["macro_rmse"] >= engression_metrics["macro_rmse"]:
        failure_reasons.append("macro RMSE did not improve on Engression")
    if not regret_safe:
        failure_reasons.append("maximum physical-unit regret exceeded 2%")

    grade_coverage_by_unit = {}
    for unit in np.unique(test["groups"]):
        mask = test["groups"] == unit
        grade_coverage_by_unit[str(unit)] = {
            "rows": int(np.sum(mask)),
            "fallback_rows": int(np.sum(mask & outside)),
            "fallback_fraction": float(np.mean(outside[mask])),
        }
    total_seconds = time.monotonic() - started
    result = {
        "schema": "ncmapss-ds03-nested-gcie-v5-development-result",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "model_name": "Grade-CVaR Implicit Expert (GCIE)",
        "development_status": frozen_payload["development_status"],
        "verdict": {
            "accepted": accepted,
            "criteria": {
                "pooled_r2_strictly_above_engression": True,
                "macro_rmse_strictly_below_engression": True,
                "maximum_unit_regret_vs_direct_at_most": MAXIMUM_WORST_REGRET,
            },
            "failure_reasons": failure_reasons,
        },
        "selection_sha256": freeze_sha256,
        "selection": frozen_payload,
        "selected_global_alpha": gate.alpha,
        "observed_source_grade_max": gate.source_grade_max,
        "grade_fallback_coverage": {
            "test_rows": int(len(test["y"])),
            "fallback_rows": int(np.sum(outside)),
            "fallback_fraction": float(np.mean(outside)),
            "inside_rows": int(np.sum(~outside)),
            "by_physical_unit": grade_coverage_by_unit,
            "outside_rows_exact_fallback": True,
        },
        "test": {
            "nested_gcie": gcie_metrics,
            "direct_fallback": direct_metrics,
            "engression_external": engression_metrics,
        },
        "comparison": {
            "gcie_minus_engression_pooled_r2": (
                gcie_metrics["pooled_r2"]
                - engression_metrics["pooled_r2"]
            ),
            "gcie_minus_engression_macro_rmse": (
                gcie_metrics["macro_rmse"]
                - engression_metrics["macro_rmse"]
            ),
            "gcie_minus_direct_pooled_r2": (
                gcie_metrics["pooled_r2"] - direct_metrics["pooled_r2"]
            ),
            "gcie_minus_direct_macro_rmse": (
                gcie_metrics["macro_rmse"] - direct_metrics["macro_rmse"]
            ),
        },
        "engression_external_audit": {
            **engression_audit,
            "artifact_root": str(ENGRESSION.relative_to(ROOT)),
        },
        "split_audit": {
            **frozen_payload["split_audit"],
            "test": _split_audit(test),
        },
        "information_contract": {
            **frozen_payload["information_contract"],
            "test_evaluations": 1,
            "test_load_attempts_in_artifact_lineage": (
                recovery_audit["test_load_attempts_in_artifact_lineage"]
                if recovery_audit is not None
                else 1
            ),
            "test_loaded_after_selection_sha256": freeze_sha256,
            "historically_revealed_test_acknowledged": True,
        },
        "timing_seconds": {
            **frozen_payload["timing_seconds"],
            "test_load_predict_score": time.monotonic() - test_started,
            "total": total_seconds,
        },
    }
    np.savez_compressed(
        args.out / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        support_grade=test_grades,
        outside_source_grade=outside,
        direct_samples=direct_samples,
        candidate_samples=candidate_samples,
        gated_samples=output_samples,
        prediction=gated_mean,
        selected_alpha=np.asarray(gate.alpha),
    )
    _write_json(args.out / "results.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
