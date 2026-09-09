# NASA PCoE second battery cohort — frozen protocol v1

**Status:** written before downloading the listed numeric MAT files or inspecting their capacities.

## Motivation

This is a small real run-to-failure battery cohort with an observable capacity health coordinate. It is a concept-aligned test for PP's strict late-capacity extrapolation, not a test selected because any model's outcome is known.

## Source and fixed candidate files

Source mirror: `https://github.com/MullaAhmed/Predicting-RUL-for-EV-Battery`, which mirrors NASA PCoE battery aging MAT files. Download **only** these eleven files: `B0029`, `B0030`, `B0031`, `B0032`, `B0046`, `B0047`, `B0048`, `B0053`, `B0054`, `B0055`, `B0056`.

Files are acquired through raw GitHub URLs, one file at a time; the raw-file SHA-256 manifest is saved before parsing.

## Eligibility and split

1. Parse discharge cycles only. A cell is eligible only when it has at least 40 finite discharge-capacity observations and crosses 70% of its median first-five-cycle capacity.
2. Eligible IDs retain the fixed order above. Do not replace an ineligible ID.
3. First 60% of eligible IDs are train, next 20% validation, remaining 20% test. If fewer than 10 cells or fewer than 2 test cells remain, stop as inconclusive.
4. A row uses only capacity up to its current cycle, its causal recent rate, cycle index and constant operating metadata. Its target is remaining cycle count to the observed 70% crossing.
5. Train rows lie above the minimum train-side capacity boundary. Validation and test rows must lie strictly below that actual train boundary. If either outer split is not 100% outside support, stop as infeasible.

## Models and selection

Use the frozen PP-X executor and a matched plain MLP, five seeds 42–46. Select learning rate, width, nonlinear trust, and early stopping only on validation RMSE using the same finite grid for both arms. No test label, feature, split, boundary, or candidate may be changed after the first result is written.

## Report

Report pooled R², unit-macro R², RMSE, MAE, per-cell metrics, five individual seed metrics, and paired cell bootstrap. This is a model-level external cohort; it does **not** itself establish a per-unit routing gate because the test cohort has fewer than ten independent units.
