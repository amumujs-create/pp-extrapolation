# NASA PHM 2019 lap-joint crack small-cohort protocol

Frozen before downloading or inspecting `PHMDC2019_Data.zip`.

## Source

- NASA Open Data: *Fatigue Crack Growth in Aluminum Lap Joint*.
- Landing page: https://data.nasa.gov/dataset/fatigue-crack-growth-in-aluminum-lap-joint
- Resource: `PHMDC2019_Data.zip` (HTTP range audit: 1,887,795 bytes).
- The NASA description states that aluminum lap-joint specimens were fatigue
  tested, Lamb-wave signals were recorded at multiple fatigue-cycle points, and
  optical surface-crack lengths are the ground truth.  It also states that an
  official training/validation split is supplied.

This dataset has not been used in the PP development table.  It is a small
non-battery structural-degradation cohort.

## Locked data routing

1. Preserve the official train/validation designation in the archive.
2. Official training specimens are development data.  If at least three are
   present, the last alphanumeric training specimen is validation and the rest
   are model-training specimens.
3. Official validation specimens are the locked external test set.
4. If official validation crack-length truth is absent, do not reconstruct or
   search for hidden labels; report `inconclusive`.
5. Hash every cleaned specimen trajectory.  Exact duplicates across routes make
   the evaluation `inconclusive`.

## RUL construction

- Ordered health coordinate: optical crack length.
- The common failure boundary is the minimum terminal crack length across the
  model-training specimens only.
- Each labelled specimen is truncated at its first boundary crossing.
- Target RUL is fatigue cycles remaining to that crossing.
- A test specimen must cross the frozen development boundary and provide at
  least 10 pre-boundary observations; otherwise it is excluded by the fixed
  eligibility rule.  Fewer than two eligible test specimens makes the cohort
  `inconclusive` for external confirmation.

## Causal inputs

The compact experiment first uses the optical crack-length trajectory only:
current crack length, boundary margin, elapsed cycles, causal finite-difference
growth rates, and causal rolling/expanding summaries.  This avoids downloading
or training on large image representations and directly tests PP's structural
RUL prior.  Future crack length, final lifetime, specimen ID and test labels are
never model inputs.

If the archive contains only sparse optical truth paired with Lamb-wave
features, those publisher-provided numeric features may be used only when their
time alignment is explicit.  Feature choices are not changed after test scores
are calculated.

## Models and metrics

- Primary: log-boundary-quotient PP frozen after Ferrara development.
- The Ferrara vibration-onset gate is disabled because its signal contract does
  not match Lamb-wave/crack-length data.
- Comparators: affine-only log quotient, boundary-rate, Ridge and matched plain
  MLP.
- Seeds: 42--46; primary prediction is their mean.
- Hyperparameters and stopping epochs use only the held-out development
  validation specimen.
- Test rows are the final 30% before the boundary crossing, with causal history.
- Report pooled R², unit-macro R², RMSE, MAE, per-unit metrics, seed dispersion,
  and ordered-coordinate hull distance.

## Success rule

An external success requires at least two eligible test specimens, pooled R²
greater than zero, PP RMSE lower than matched MLP RMSE, and no provenance or
duplicate gate failure.  A result that misses any condition remains in the
repository as failure or inconclusive evidence.
