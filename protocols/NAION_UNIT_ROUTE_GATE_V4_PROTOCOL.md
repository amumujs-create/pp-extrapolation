# Na-ion prospective unit-route gate (v4)

Version 4 is frozen after group-3 outcome analysis and before any group-4 file is
downloaded or inspected. Group 3 showed that a dataset-level decision can mix one
PP-compatible cell with one incompatible cell. The new decision is therefore
made per test unit.

## Locked cohort

- BatteryLife raw Na-ion group 4 files 1--7 and `270040-4-8-41.csv`.
- `270040-4-8-40.csv` is excluded before download because the official
  BatteryLife preprocessor declares it a problematic cell.
- Lexicographic order: first four train, next two validation, final two test.
- All preprocessing, causal rolling-tail split, models, seeds, and survival
  projection are fixed as in the v3 protocol.

## Frozen unit-level route

The global PP route must pass validation relative-MSE gain >=2%, projected seed
disagreement <=0.25, and minimum unit counts. Each test unit is then independently
approved only when both are true:

1. robust prefix distance to the nearest train unit is <=3.0;
2. robust prefix distance to the nearest validation unit is <=1.5.

Descriptors, train-based scaling, and distance normalization are unchanged. The
validation-neighborhood condition requires the new unit to resemble a unit on
which PP's relative skill was actually observed. All decisions and prefix hashes
are written before test RUL targets are constructed.

Primary evaluation reports PP coverage, selective pooled R2 over PP-approved
units, route accuracy by unit (PP if its projected RMSE is no worse than projected
MLP and its R2 is positive), and deployment pooled R2 obtained by PP on approved
units and MLP on rejected units. No threshold may change after group-4 scoring.

