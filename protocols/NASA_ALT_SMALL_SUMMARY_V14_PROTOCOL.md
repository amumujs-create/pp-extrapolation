# NASA ALT small-summary PP-X v1.4 protocol

**Frozen:** 2026-09-10, 박진서, before extracting any capacity outcome.

## Source and small-data contract

- Source: NASA/PML-UCF *Accelerated Life Testing Dataset for Lithium-Ion
  Batteries with Constant and Variable Loading Conditions* v0.0.1.
- Local archive SHA-256:
  `89209acddbad47506d781698fc51ebe9d7cdb8968667698f63e63aa730981d21`.
- The 513 MB telemetry ZIP is reduced once to one reference-discharge capacity
  per pack-day. The scored summary, not the 3.4 GB expanded telemetry, is the
  model input and must be saved separately.
- Include all 26 packs: 15 regular, 8 recommissioned load-switching, and 3
  second-life packs. No pack may be replaced after eligibility is known.

For rows with `mode=-1` and `mission_type=0`, integrate trapezoidal
`current_load` over adjacent samples with `0 < dt <= 5 s`. A reference cycle
requires at least 600 valid samples. Sort cycles by first relative time.

## Frozen unit split

The split was generated from folder names and pack IDs only with NumPy
`default_rng(20260910)` independently inside each stratum.

- train regular: 21, 40, 01, 51, 30, 41, 11, 50
- validation regular: 31, 52, 00, 22
- test regular: 23, 10, 20
- train recommissioned: 24, 02, 25, 32
- validation recommissioned: 53, 33
- test recommissioned: 03, 12
- train second-life: 54
- validation second-life: 36
- test second-life: 13

Thus train/validation/test contain 13/7/6 physical packs before eligibility.

## Target and strict tail

- Normalize capacity by the median of the first five valid reference cycles.
- Define EOL as the first of two consecutive capacities at or below 80%.
- Require at least 10 reference cycles and crossing index at least 8.
- Use a causal five-cycle history. Features are current health, slopes over
  lags 1/3/4, trailing mean/std, and reference-cycle index. Folder/category is
  not a feature.
- Target is reference cycles remaining to EOL.
- Set the outer boundary to the 25th percentile of eligible train health.
  Fit rows are at or above that boundary; validation/test rows are strictly
  below the realized minimum train health.
- If fewer than four test packs or 20 pooled strict-tail rows remain, stop as
  inconclusive.

## Frozen models and router

- Fixed architecture bank: widths 16/32, learning rate `1e-3`, weight decay
  2.0, seeds 42--46.
- Direct-residual trust bank: `0,.02,.05,.10,.20,.40`; trust zero is the
  baseline portfolio.
- Fit two train-only statistical regimes. A validation regime with fewer than
  three physical packs must use baseline. This is the preregistered
  low-volume adaptation; assembled raw validation regret remains constrained.
- Regime route alpha is in `0,.05,...,.25`. Final raw physical-pack mean,
  worst-20% CVaR, and maximum regret caps are 2%/5%/10%; failure gives exact
  baseline.
- Test labels cannot affect fitting, regime assignment, expert choice, alpha,
  or fallback.

## Success and reporting

Primary success requires all of:

1. admissibility passes;
2. pooled test RMSE is no worse than the baseline portfolio;
3. test raw mean/CVaR/maximum regret satisfy 2%/5%/10%;
4. pooled R-squared is positive.

Report pooled and physical-pack metrics, each source stratum separately,
regime occupancy, selected trusts/alphas, exact fallback error, and extracted
summary size. Because architecture development already used other opened
datasets, this is an external one-shot test of v1.4, not a claim that the NASA
archive itself was never inspected for metadata.
