# ISU–ILCC 250 mAh cohort — frozen PP-X v1.2 protocol

**Frozen:** 2026-09-10, 박진서. Written after reading only the public landing
metadata, README, paper methods, and file list; before downloading or opening
`capacity_fade.zip`.

## Cohort

- ISU–ILCC Battery Aging Dataset v2, DOI `10.25380/iastate.22582234.v2`.
- 502030 NMC/graphite pouch cells, rated capacity **250 mAh**, 30 °C.
- Public release: 238 cells across 63 charge-rate, discharge-rate, and DoD
  conditions. Associated paper reports 225 completed cells.
- The authors define lifetime by the first RPT capacity below **200 mAh**
  (80% rated capacity).
- Payload is the authors' 140,256-byte `capacity_fade.zip`, containing
  two-column per-cell time-on-test and remaining-capacity trajectories.
- Workspace audit found no prior use of DOI 22582234 or ISU–ILCC. This is
  independent of both UConn candidate attempts.

## Eligibility and split

1. Use all listed valid cells with at least 10 finite ordered capacity
   observations and an observed first crossing at or below 0.200 Ah.
2. No exclusion or replacement based on lifetime, protocol, curve shape, or
   score. Report duplicate-time/schema/readability exclusions.
3. Parse IDs as `(group number, cell number)` and sort numerically. First 60%
   train, next 20% validation, final 20% test. This intentionally evaluates
   later condition groups rather than randomizing conditions across splits.
4. Stop below 100 eligible cells, 20 validation cells, or 20 test cells.

## Target, features, and strict extrapolation

- Endpoint: first observed capacity `<=0.200 Ah`.
- Target: remaining time-on-test in days to endpoint.
- Causal features: capacity/rated capacity, 1/3/5-observation slopes in
  health per day, five-observation causal mean/std, and elapsed days.
- Unit/group identity and held-out batch statistics are prohibited inputs.
- Freeze the health cutoff at the 25th percentile of pre-event training rows.
  Fit only rows strictly above it. Realized boundary is the minimum retained
  train health. Score validation/test rows strictly below that boundary.
- Stop as infeasible if validation or test has fewer than 100 scored rows or
  less than 100% strict tail coverage.

## PP-X v1.2 and matched fallback

Grid: widths `{16,32}`, learning rates `{5e-4,1e-3}`, weight decay `2.0`,
trust `{0,.02,.05,.1,.2,.4}`, selection seeds `42–44`, final seeds `42–46`,
300 epochs, patience 50.

A positive-trust prior route is approved only when all validation checks pass:

1. pooled RMSE improvement over exact matched MLP is at least 2%;
2. at least 60% of validation cells improve;
3. worst-cell candidate/fallback RMSE ratio is at most 1.10;
4. physical-cell bootstrap 95% lower bound of fallback-minus-candidate RMSE
   is positive.

Otherwise trust is zero. Trust-zero predictions must replay the independently
fit matched MLP within maximum absolute error `1e-5`.

Report pooled, cell-macro and per-cell R²/RMSE/MAE, five seeds, cell bootstrap,
split IDs, hull, hashes and every validation candidate. Validation R² `<=0`
makes the numerical test result uncertified. Model success requires test
pooled R² `>0` and selected RMSE no worse than matched MLP. A PP prior
generalization claim additionally requires positive selected trust.

No rule may change after opening `capacity_fade.zip`.
