# External-cohort four-model comparison protocol

## Scope

This retrospective benchmark compares plain MLP, official Engression, PP-X,
and CDCR-PPX on five previously opened external settings:

- Axial-fan `1P_8F`, `4P_1F`, and `4P_8F`;
- MATWI tool life;
- Misata machine degradation.

Every comparison reuses the original unit split, causal features, target, and
test rows. Test labels are used only for final metrics. Because all outcomes
were opened before this benchmark, no result is prospective confirmation.

## Frozen model inputs

- Axial-fan PP-X and MLP are the five-seed endpoint-matched tail-gate
  development predictions. This route was developed after the untouched test
  failure and is post-test evidence.
- MATWI uses the five-seed PP core and matched MLP predictions frozen before
  untouched scoring. No separate PP-X executor was approved for this
  label-only cohort, so the PP core is the applicable PP-X route.
- Misata uses the five-seed PP-X residual-executor predictions and the matched
  MLP predictions frozen in the original cohort artifact. The executor was
  developed after the first Misata outcome was opened.
- CDCR-PPX is the fixed nine-domain development head applied to the PP-X seed
  matrix for each setting. External labels never fit or select this head.

## Engression

Use the installed official Engression package with the same family and budget
as the main equal-budget comparison. Search hidden width `{32,64}`, learning
rate `{0.001,0.005}`, beta `{0.5,1.0}`, and `(layers, epochs)` equal to
`{(2,250),(3,500)}`. Select by validation MSE using seed 42, then refit the
selected fixed-epoch configuration on all development units for seeds 42--46.
MATWI uses its original four development-unit folds and selects by mean fold
MSE. Predictions are clipped to `[0, max development target]`.

## Metrics

Primary metrics are setting-level pooled R2 and RMSE. Report equal-setting
mean R2, worst-setting R2, number of positive-R2 settings, strict wins, ties,
mean seed R2, minimum seed R2, and mean within-setting seed R2 standard
deviation. Do not concatenate heterogeneous target scales into one raw pooled
R2.

## Interpretation firewall

The five settings mix untouched base-model evidence and post-test PP-X
development. They are a stress-test audit, not a new external confirmation.
Axial-fan configuration rows are correlated views of one public cohort and
must not be described as three independent datasets.
