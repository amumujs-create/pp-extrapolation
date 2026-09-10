# SNL / Sandia Battery Archive cohort — frozen PP-X v1 protocol

**Status:** written before downloading `SNL.zip` or inspecting cell identifiers, capacities, lifetimes, or any model score. This protocol is used only if `UL_PUR_PPX_V1_PROTOCOL.md` stops as inconclusive for sample size or archive unreadability. It is not a backup after a negative UL-PUR score.

## Motivation

SNL is the next unused BatteryLife processed Li-ion archive in this workspace after UL-PUR. It is listed here so a sample-size failure on UL-PUR cannot be repaired by choosing SNL after seeing UL-PUR labels.

## Source and acquisition

- Distribution: BatteryLife processed `SNL.zip`.
- Preferred record: Zenodo `14934405`, file `SNL.zip`.
- Fallback: Hugging Face `Hongwxx/BatteryLife_processed`.
- Download the zip once. Save SHA-256 of the archive before parsing pickle contents.

## Eligibility, split, models, and report

Identical to `UL_PUR_PPX_V1_PROTOCOL.md`: 20 finite discharge capacities, first crossing of 80% of the median first-five-cycle capacity, lexicographic 60/20/20, train above the actual train capacity minimum, scored validation/test strictly below it, frozen PP-X v1.0 executor versus matched MLP, widths `{16,32}`, learning rates `{5e-4,1e-3}`, seeds `42–46`. Inconclusive if fewer than 10 eligible cells or fewer than 2 test cells. Infeasible if a scored outer split is empty or not 100% outside train support.

Confirmatory success requires PP-X ensemble pooled R² > 0 and PP-X ensemble pooled RMSE lower than the matched MLP. After the first SNL test metric is written, later changes are development evidence.
