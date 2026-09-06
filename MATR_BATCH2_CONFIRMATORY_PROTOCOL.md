# MATR batch2 sealed confirmatory protocol

Status: frozen before downloading or deserializing MATR batch2. Earlier PAE records
explicitly report zero batch2 deserializations; batch1+3 were development data.

## Fixed unit split

Read the 48 raw batch2 cells in their stored order. Use cells 0--29 for train, 30--38
for validation, and 39--47 for final test. If fewer than 48 cells exist, abort. Do not
remove, reorder, or replace cells after inspecting lifetime or capacity.

## Tail construction

Use per-cycle `QDischarge` and a causal nonnegative capacity-loss rate. Define the
cutoff as the 25th percentile of endpoint capacity among train eight-cycle windows.
Retain train windows strictly above the cutoff and validation/test windows strictly
below the minimum retained train capacity. Require validation and test to be 100%
outside the one-dimensional train capacity convex hull and contain at least 20 rows.

Inputs are last, mean, and endpoint change of capacity and causal loss rate over eight
cycles. Cycle index, charging policy, cell identity, total lifetime, and future values
are excluded. Target is remaining measured cycles to the stored end of life.

## Models, gate, and success

Fit Ridge/affine, plain NN, PP, and support-PP with seeds 42--46. Use the frozen
distance-shell gate from `f6eb4c3`. Serialize the gate decision before materializing
test targets.

Confirmatory predictive success requires all of:

1. nonzero gate prediction coverage;
2. selective PP pooled R² > 0;
3. selective PP RMSE below selective Ridge RMSE;
4. full-test PP ensemble pooled R² above Ridge pooled R²;
5. no protocol or threshold change after opening batch2.

Report pooled R²/RMSE, unit-macro R², seed dispersion, prediction coverage, and
selective regret even when the criteria fail. The raw source is the 124-cell dataset
of Severson et al., distributed by TRI under CC BY 4.0.
