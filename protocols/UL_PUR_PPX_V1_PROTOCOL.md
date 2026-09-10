# UL-PUR Battery Archive cohort — frozen PP-X v1 protocol

**Status:** written before downloading `UL_PUR.zip` or inspecting cell identifiers, capacities, lifetimes, or any model score.

## Motivation

NASA PCoE second (B0029–B0056) was inconclusive under its frozen 70% first-five-median crossing rule: zero eligible cells, and ineligible IDs were not replaced. The next concept-aligned test is a different public Li-ion archive with an observable capacity health coordinate. UL-PUR is unused in this PP workspace. It is selected for source and size, not because any model outcome is known.

## Source and acquisition

- Distribution: BatteryLife processed `UL_PUR.zip` (same family as the already-used CALCE/HNEI processed release).
- Preferred record: Zenodo `14934405`, file `UL_PUR.zip`.
- Fallback if that file URL is unavailable: the matching `UL_PUR.zip` on Hugging Face `Hongwxx/BatteryLife_processed`.
- Download the zip once. Save SHA-256 of the archive before parsing pickle contents.
- Do not download SNL, CALB, Tongji, Stanford, or any other BatteryLife zip in this run unless this protocol stops as inconclusive for sample size.

## Eligibility and split

1. Parse every readable cell pickle. A cell is eligible only when it has at least 20 finite discharge-capacity observations and crosses 80% of its median first-five-cycle capacity. The first such crossing is the event. This is the BatteryLife-style 80% boundary, not a test-chosen cutoff.
2. Eligible IDs are sorted lexicographically. Do not drop, replace, or reorder an ID because of lifetime, fade shape, or a preliminary score.
3. First 60% of eligible IDs are train, next 20% validation, remaining 20% test. If fewer than 10 eligible cells or fewer than 2 test cells remain, stop as inconclusive. Do not relax the 80% rule.
4. A row uses only capacity up to its current cycle, its causal recent rate, cycle index, and any constant operating metadata present in the processed record. Its target is remaining cycle count to the observed 80% crossing.
5. Train rows lie above the minimum train-side capacity boundary. Validation and test rows used for selection or scoring must lie strictly below that actual train boundary. If either scored outer split is empty or not 100% outside that support, stop as infeasible.

## Models and selection

Use the frozen PP-X v1.0 core (affine prior + residual) and a matched plain MLP. Residual executor is chosen by the frozen `select_residual_executor` policy from unbounded (`residual_decay=0`) versus bounded (`residual_decay=0.3`) validation loss. Select learning rate, width, and early stopping only on validation RMSE using the same finite grid for both arms: widths `{16,32}`, learning rates `{5e-4,1e-3}`, seeds `42–46`. No Oxford-style scale patch, dual-scale add-on, or post-test feature is allowed.

## Report

Report pooled R², unit-macro R², RMSE, MAE, per-cell metrics, five individual seed metrics, and paired cell bootstrap. Confirmatory success requires PP-X ensemble pooled R² > 0 and PP-X ensemble pooled RMSE lower than the matched MLP. This is a model-level external cohort. It does not itself establish a per-unit routing gate if the test split has fewer than ten independent units.

After the first test metric is written, any change to eligibility, split, boundary, features, or the model grid converts later UL-PUR numbers to development evidence.
