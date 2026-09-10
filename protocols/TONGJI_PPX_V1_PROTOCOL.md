# Tongji BatteryLife cohort — frozen PP-X v1 protocol

**Status:** written before downloading `Tongji.zip` or inspecting cell identifiers, capacities, lifetimes, or any model score. Used only if `CALB_PPX_V1_PROTOCOL.md` stops as inconclusive for sample size or archive unreadability. Not a backup after a negative CALB score.

## Source and acquisition

- Distribution: BatteryLife processed `Tongji.zip`.
- Preferred record: Zenodo `14934405`, file `Tongji.zip`.
- Fallback: Hugging Face `Hongwxx/BatteryLife_processed`.
- Download the zip once. Save SHA-256 before parsing.

## Eligibility, split, models, and report

Identical to `CALB_PPX_V1_PROTOCOL.md`: 20 finite discharge capacities, first 80% first-five-median crossing, lexicographic 60/20/20, **normalized health** as the one-dimensional hull coordinate, train on or above the actual train health minimum, scored validation/test strictly below it, frozen PP-X v1.0 versus matched MLP, widths `{16,32}`, learning rates `{5e-4,1e-3}`, seeds `42–46`.
