# Additional real-dataset protocol

This extension adds three locally available real battery degradation datasets:
Sunwoda, RWTH, and MICH. It tests whether PP's tail-extrapolation behavior carries
to new physical units and datasets beyond the benchmarks already reported.

## Frozen split and inputs

- Use the existing unit-disjoint splits produced by `pae_boundary_realdata.prepare_dataset`.
- The loader provides early-life train/validation units and strict late-health test
  units. Before fitting, define an internal health cutoff from the 25th percentile
  of train-window endpoint health: retain train rows above it and validation rows
  below it. This makes validation a held-out-unit pseudo-tail without test access.
- Convert each causal 8-step window to the same generic summaries: last value, mean,
  and endpoint change for health and rate, plus normalized cycle. No battery equation
  or test-label-derived feature is inserted.
- Assert that every retained validation and test row is below the one-dimensional
  convex hull of retained training health before fitting.

## Frozen comparison

- Ridge: validation-selected regularization on the same inputs.
- Plain NN: the nonlinear branch without an affine tail prior.
- PP: affine tail plus learned nonlinear correction.
- Support-adaptive PP: PP with validation-selected correction decay outside support.
- Seeds: 42, 43, 44, 45, 46.
- Primary metric: raw pooled R². Also report RMSE and unit-macro R².

Model selection may use training and validation labels only. Test labels are opened
once after all choices are fixed. Results are reported even when negative.

## Evidence status

These datasets have previously been inspected during PAE development. Therefore this
is a cross-model replay and coverage stress test, not a globally untouched prospective
validation. A later untouched cohort is required for the strongest paper claim.
