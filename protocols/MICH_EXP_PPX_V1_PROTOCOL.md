# MICH_EXP BatteryLife cohort — frozen PP-X v1 protocol

**Status:** written before downloading `MICH_EXP.zip` or inspecting cell identifiers, capacities, lifetimes, or any model score. Used only if `STANFORD_PPX_V1_PROTOCOL.md` stops as inconclusive for sample size or archive unreadability. Not a backup after a negative Stanford score.

## Source and extra reuse gate

- Distribution: BatteryLife processed `MICH_EXP.zip`.
- Preferred record: Zenodo `14934405`.
- The main-table MICH archive (`MICH.zip`) is PP-seen. Before scoring, abort as reuse if any parsed `cell_id` equals a main-table MICH cell identifier from `results/bq_dual_scale_final_replay_v1/results.json`.

## Eligibility, split, models, and report

Identical to `STANFORD_PPX_V1_PROTOCOL.md`: 20 finite discharge capacities, first 80% first-five-median crossing, lexicographic 60/20/20, normalized health, **25th-percentile train health cutoff**, train strictly above that cutoff, scored validation/test strictly below the actual retained-train minimum, frozen PP-X v1.0 versus matched MLP, widths `{16,32}`, learning rates `{5e-4,1e-3}`, seeds `42–46`.
