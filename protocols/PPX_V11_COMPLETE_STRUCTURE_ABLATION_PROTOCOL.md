# PP-X v1.1 complete structural ablation contract

**Frozen:** 2026-09-10, 박진서. Retrospective mechanism analysis; not a new
confirmatory experiment. Existing v1.0 ablations remain unchanged.

## Cohorts

1. Stanford BatteryLife: post-test development cohort used to create v1.1.
2. ISU–ILCC 250 mAh: later external cohort, reused here only for replication.

Use each cohort's already frozen split, eligibility, features and strict-tail
boundary. Test labels may measure retrospective arms but may not select them.

## Full v1.1 arm set

- Matched fallback: trust `0`.
- Prior dose: trust `.02, .05, .10, .20, .40`.
- Network budget at every dose: width `{16,32}` × learning rate
  `{5e-4,1e-3}`, weight decay `2.0`.
- Seeds: `42–46`.
- Gate 2×2:
  - 2% relative validation-RMSE margin off/on;
  - physical-unit bootstrap 95% lower bound off/on.
- Exact fallback audit: gated prediction versus matched trust-zero prediction.
- Seed stability: per-seed pooled R²/RMSE for matched MLP, validation-best
  always-on prior and full v1.1 gate.

For every trust dose, architecture is selected by validation ensemble RMSE.
The always-on arm is the validation-best positive-trust arm. Each gate arm
either returns that same arm or the validation-selected trust-zero fallback.
No test score participates in route selection.

Bootstrap uses 20,000 physical-unit resamples and seed 20260910. A positive
lower bound means the 2.5th percentile of
`fallback RMSE − candidate RMSE` is above zero.

## Outputs and claims

Report all 24 architecture×trust validation/test rows per cohort, the trust
dose curve, gate 2×2 decision and outcome, per-seed stability, unit wins,
bootstrap interval, and exact replay error.

This ablation tests all configurable structural choices in the implemented
v1.1 safety continuation. It does not enumerate arbitrary new priors, feature
sets, thresholds outside the frozen 2%/95% rule, or v1.2's later worst-unit
extensions.
