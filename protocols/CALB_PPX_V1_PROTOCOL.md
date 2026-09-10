# CALB BatteryLife cohort — frozen PP-X v1 protocol

**Status:** written before downloading `CALB.zip` or inspecting cell identifiers, capacities, lifetimes, or any model score.

## Motivation

Three pre-registered archives were already closed without a PP-X score:

- NASA PCoE second: inconclusive (0 eligible 70% crossings; IDs not replaced).
- UL-PUR: inconclusive (2 cells, 0 eligible).
- SNL: infeasible. Lexicographic eligible IDs mix LFP (~1 Ah) with NCA/NMC (~3 Ah), so a raw-Ah train minimum leaves every NCA/NMC validation/test row inside support.

SNL is not re-run. The next unused BatteryLife processed archive is CALB. The one-dimensional hull coordinate is **normalized health**, not raw ampere-hours. This choice is locked here from the SNL support audit, not from a CALB or SNL test R².

## Source and acquisition

- Distribution: BatteryLife processed `CALB.zip`.
- Preferred record: Zenodo `14934405`, file `CALB.zip`.
- Fallback: Hugging Face `Hongwxx/BatteryLife_processed`.
- Download the zip once. Save SHA-256 of the archive before parsing pickle contents.
- Do not download Tongji or any later archive unless this protocol stops as inconclusive for sample size or archive unreadability.

## Eligibility and split

1. Parse every readable cell pickle. A cell is eligible only when it has at least 20 finite discharge-capacity observations and crosses 80% of its median first-five-cycle capacity. The first such crossing is the event.
2. Eligible IDs are sorted lexicographically. Do not drop, replace, or reorder an ID because of lifetime, chemistry, fade shape, or a preliminary score.
3. First 60% of eligible IDs are train, next 20% validation, remaining 20% test. If fewer than 10 eligible cells or fewer than 2 test cells remain, stop as inconclusive. Do not relax the 80% rule.
4. Health at a cycle is that cycle's capacity divided by the cell's median first-five-cycle capacity. A row uses health, causal recent health rate, cycle index, and any constant operating metadata. Its target is remaining cycle count to the observed 80% crossing.
5. The train-side boundary is the minimum health among train rows before their 80% crossing. Train rows lie on or above that boundary. Scored validation and test rows must lie strictly below it. If either scored outer split is empty or not 100% outside that support, stop as infeasible.

## Models and selection

Frozen PP-X v1.0 core versus matched plain MLP. Residual executor from `select_residual_executor` on unbounded (`residual_decay=0`) versus bounded (`residual_decay=0.3`) validation loss. Widths `{16,32}`, learning rates `{5e-4,1e-3}`, seeds `42–46`. No Oxford-style scale patch and no SNL re-run.

## Report

Pooled R², unit-macro R², RMSE, MAE, per-cell metrics, five seeds, paired cell bootstrap. Confirmatory success requires PP-X ensemble pooled R² > 0 and PP-X ensemble pooled RMSE lower than the matched MLP. After the first CALB test metric is written, later changes are development evidence.
