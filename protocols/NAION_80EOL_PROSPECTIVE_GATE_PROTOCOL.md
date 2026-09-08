# Prospective Na-ion 80%-EOL applicability protocol

This protocol is frozen before downloading any test file. Earlier Na-ion group
experiments incorrectly treated recording termination as failure; they are audit
evidence only. This protocol follows BatteryLife's official life-label rule.

## Failure and development data

- Nominal capacity is 1.0 Ah. EOL is the first cycle whose maximum discharge
  capacity is <=0.80 Ah. Cells with EOL <40 cycles are excluded.
- Development train cells are eligible cells in downloaded groups 1--3.
- Development validation cells are eligible cells in downloaded group 4.
- Every trajectory is truncated at EOL; recording termination is never a target.

## Untouched test cohort

The exact files are fixed from BatteryLife's published NAion split list:
`270040-6-2-30.csv`, `270040-6-6-26.csv`, `270040-6-8-24.csv`,
`270040-7-1-23.csv`, and `270040-8-5-16.csv`. None was downloaded or inspected
before this protocol. A test cell that does not reach 80% EOL is reported as
right-censored and excluded under this predeclared rule.

## Model, split, and gate

- The observation boundary is 60% of median train EOL, computed from train only.
- PP/plain MLP, causal features, seeds 42--46, and survival-support projection
  are unchanged. Test projection uses maximum known train/validation EOL.
- Before test EOL is calculated, the global gate requires projected validation
  MSE gain >=2%, seed disagreement <=0.25, >=4 train cells, and >=2 validation
  cells. Each test unit additionally requires nearest-train prefix distance <=3.0
  and nearest-validation prefix distance <=1.5.
- Prefix descriptors use observations only through the train-defined boundary.
  The decision and prefix hash are saved before test EOL materialization.

Success requires nonzero PP coverage, correct PP/MLP routing on every evaluable
test unit, positive selective PP pooled R2, and deployment-route pooled R2 no
worse than matched projected MLP. All exclusions, cells, seeds, and per-unit
metrics are reported.

