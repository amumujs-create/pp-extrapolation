# Stanford BatteryLife cohort — frozen PP-X v1 protocol

**Status:** written before downloading `Stanford.zip` or inspecting cell identifiers, capacities, lifetimes, or any model score.

## Motivation

The previous unused-archive queue (NASA PCoE second, UL-PUR, SNL, CALB, Tongji) closed without a PP-X score. Tongji showed that a train boundary equal to the minimum pre-80% health is empty by construction. Those archives are not re-run. The next unused BatteryLife processed archive is Stanford. The one-dimensional hull uses **normalized health** and the lab's **25th-percentile train-endpoint cutoff**, locked here before download.

This is PP model-level external evidence. PAE notes mention a different Stanford-2 screen; that fact, if it applies only to `Stanford_2.zip`, is not used to choose or drop cells here.

## Source and acquisition

- Distribution: BatteryLife processed `Stanford.zip` (not `Stanford_2.zip`).
- Preferred record: Zenodo `14934405`, file `Stanford.zip`.
- Fallback: Hugging Face `Hongwxx/BatteryLife_processed`.
- Download the zip once. Save SHA-256 of the archive before parsing pickle contents.
- Do not download `MICH_EXP.zip` unless this protocol stops as inconclusive for sample size or archive unreadability.

## Eligibility and split

1. Parse every readable cell pickle. A cell is eligible only when it has at least 20 finite discharge-capacity observations and crosses 80% of its median first-five-cycle capacity. The first such crossing is the event.
2. Eligible IDs are sorted lexicographically. Do not drop, replace, or reorder an ID because of lifetime, chemistry, fade shape, or a preliminary score.
3. First 60% of eligible IDs are train, next 20% validation, remaining 20% test. If fewer than 10 eligible cells or fewer than 2 test cells remain, stop as inconclusive. Do not relax the 80% rule.
4. Health at a cycle is that cycle's capacity divided by the cell's median first-five-cycle capacity. A row uses health, causal recent health rate, cycle index, and any constant operating metadata. Its target is remaining cycle count to the observed 80% crossing.
5. From train rows before the 80% crossing, set `cutoff` to the 25th percentile of their health. Retain train rows **strictly above** `cutoff`. The train-side boundary is then the minimum health among those retained rows. Scored validation and test rows must lie strictly below that actual train boundary. If either scored outer split is empty or not 100% outside that support, stop as infeasible.

## Models and selection

Frozen PP-X v1.0 core versus matched plain MLP. Residual executor from `select_residual_executor` on unbounded (`residual_decay=0`) versus bounded (`residual_decay=0.3`) validation loss. Widths `{16,32}`, learning rates `{5e-4,1e-3}`, seeds `42–46`. No Oxford-style scale patch. No Tongji/SNL/CALB re-run.

## Report

Pooled R², unit-macro R², RMSE, MAE, per-cell metrics, five seeds, paired cell bootstrap. Confirmatory success requires PP-X ensemble pooled R² > 0 and PP-X ensemble pooled RMSE lower than the matched MLP. After the first Stanford test metric is written, later changes are development evidence.
