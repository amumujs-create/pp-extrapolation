# OAIR v1.5 post-test development protocol

**Recorded:** 2026-09-10, 박진서.

This repair was designed after the absolute-output PP-X v1.4 failed the three
LED strong-extrapolation scores. It is development only and cannot replace
that frozen negative result.

## Persistence-Anchored Ordered-Axis Invariant Residual

For current health `h_t`,

`h_(t+2) = h_t + gamma * clip(r_beta(slope_1, slope_3, slope_4), +/-c)`.

- Exact baseline is causal persistence `h_t`.
- The residual excludes health level and age, the ordered coordinates along
  which extrapolation is intended.
- `r_beta` is equal-unit weighted ridge with alpha selected from
  `.1,1,10,100,1000,10000` by validation physical-unit MSE.
- Bound `c` is twice the train 95th percentile absolute two-step change.
- Choose validation mass on `0,.01,...,1` under raw unit mean/CVaR/max caps
  2%/5%/10%.
- Deployment mass is 25% of that feasible validation mass.
- Slope-space distance above the maximum unlabeled validation distance gives
  exact persistence.

Replay the identical Data_ID_2/5/6 splits and strong-origin regions frozen in
`LED_STRONG_EXTRAPOLATION_V14_PROTOCOL.md`.

Development success requires positive test R-squared, RMSE below persistence,
and raw unit caps on all three cohorts. A new score-unopened cohort is still
required for confirmation.
