# RAVEN-X TRAIN-only screen protocol

Status: design-time protocol. No RAVEN-X training or evaluation has been run under this protocol.

## Primary hypothesis

Jointly learning source-window evidence, future relation validity, and sparse relation-constrained regime transitions improves held-unit lower-health extrapolation over parameter-matched temporal attention, regime mixtures without validity, and unrestricted transition noise.

## Data boundary

- Development data: MICH TRAIN only.
- Outer splits: 3 held-unit folds x health-support cutoffs 0.6 and 0.8.
- Inner training splits: repeat 3 held-unit folds x cutoffs 0.6 and 0.8 inside each outer donor set.
- Every source unit must be disjoint from the associated query unit.
- Every source health-margin row must be strictly above every associated query row.
- MICH validation and test must not be loaded by the first-stage runner.
- Existing opened endpoints are development data, not prospective confirmation.

## Fixed minimal architecture

- Four causal relative windows: last 1/8, 1/4, 1/2, full prefix; minimum four rows.
- Three regime paths: persistence, level transition, slope-curvature transition.
- At most one future transition.
- Eight health-distance transition bins plus a no-transition atom.
- One sparse attention head per regime, shared window encoder hidden width 16.
- Sparse allowed-edge graph from at most ten observed relation features to a, b, c, and transition hazard.
- Positive piecewise log-slowness integral with exact zero at the known boundary.
- At most 5,000 trainable parameters.
- Five seeds saved individually.

## Locked controls

1. Ridge prefix-to-function distribution.
2. Parameter-matched ordinary temporal attention without regimes.
3. Regime mixture without future-validity survival.
4. Validity model with unrestricted changes to all coefficient components.
5. Relation-mask model without counterfactual window-removal supervision.
6. Independent Gaussian perturbations in place of relation-constrained transitions.
7. Point-estimated transition location.
8. Independent horizon decoder in place of coherent piecewise integration.

Full PP-X and official Engression are not required for this rejecting TRAIN screen. They become mandatory after the internal source gate passes.

## Primary scores

- Episode-average unit MSE of the predictive mean.
- Worst held-unit MSE across episodes.
- Energy score of the full mixture.
- 90% empirical interval coverage and mean width.
- Transition-neighborhood and non-transition errors reported separately.
- All rows and units remain in the denominator.

## Internal source gate

RAVEN-X advances only if all conditions hold before loading validation:

- At least 5% lower episode-average unit MSE than every locked neural control and at least 2% lower than ridge.
- Lower energy score than every locked control.
- Worst held-unit MSE no worse than ridge.
- Empirical 90% coverage between 0.85 and 0.95 and interval score below ridge.
- No decrease in the number of held units with positive R2 versus ridge.
- Relation-mask ablation worsens both unit MSE and energy score.
- At least four of five seeds improve over ridge in unit MSE.
- Synthetic falsification recovers directly changed relation edges above a fixed chance baseline and leaves protected edges unchanged within a prespecified tolerance.

No single metric substitutes for another. Failure stops this architecture version. Thresholds cannot be relaxed after observing the outer queries.

## Validation and promotion

After an internal pass, freeze architecture, seeds, checkpoints, mixture sampling, fallback thresholds, and refit procedure. Compare on validation against the complete PP-X under identical permitted inputs. Do not open test after validation failure. Later promotion follows `protocols/SUCCESSOR_EXTRAPOLATION_ACCEPTANCE_PROTOCOL.md`, including equal-condition full PP-X and Engression comparisons, coverage preservation, independent-cohort uncertainty, and prospective confirmation.

## Claims explicitly prohibited by this screen

- Causal counterfactual identification without intervention assumptions/data.
- Novelty from attention, switching dynamics, graph masks, or physics constraints individually.
- Universal regime discovery from latent states without labeled confirmation.
- Calibration guarantees from empirical coverage alone.
- DS03 or full benchmark coverage from a MICH-only result.
- Superiority over full PP-X or Engression before their locked matched evaluation.
