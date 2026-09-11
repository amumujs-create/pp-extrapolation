# N-CMAPSS DS03 PP-X v1 prospective protocol

Owner: 박진서  
Status at freeze: **test outcome not downloaded or inspected in this project**  
Paper method: PP-X Paper Method v1  
Dataset file: `N-CMAPSS_DS03-012.h5`

## Purpose

This is the first prospective replay of the already frozen PP-X paper method.
N-CMAPSS DS02-006 was used during method development. DS03-012 is a separate
15-unit cohort with a different failure mechanism and has not previously been
referenced, loaded, profiled, or evaluated by this repository.

This is an external-within-family validation, not an independent real-world
domain validation. Its result must be reported whether positive or negative.

## Source and integrity

- Source: NASA Prognostics Center of Excellence, Turbofan Engine Degradation
  Simulation Data Set 2
- Canonical filename: `N-CMAPSS_DS03-012.h5`
- Expected units: development 1–9; test 10–15
- Raw file SHA-256: recorded immediately after download and before opening HDF5
- No dataset-specific architecture or threshold change is allowed after the
  raw hash is recorded.

## Frozen information contract

- Inputs: the same observable scenario and sensor columns used by the frozen
  DS02 PP-X adapter; no health parameters, virtual ground-truth state, failure
  mode label, or test outcome.
- Target: remaining useful life in cycles.
- Ordered coordinate: flight cycle.
- Group: physical engine unit.
- History: causal windows only.
- Test-batch statistics: prohibited.

## Frozen partitions

- Source training units: DS03 development units 1–6.
- Validation units: DS03 development units 7–9.
- Prospective test units: DS03 test units 10–15.
- Training rows may be deterministically capped per unit using the existing
  DS02 adapter rule. The same cap applies to all competitors.
- No test unit can be moved to training or validation.

## PP-X selection

The candidate set is fixed before data access:

1. direct neural fallback,
2. basic causal prior-residual executor,
3. multiscale causal prior-residual executor.

Candidates are restricted and selected with PP-X Paper Method v1:

- validation gain at least 2% over fallback,
- validation physical-unit win fraction at least 60%,
- worst validation-unit RMSE ratio at most 1.10,
- deterministic tie break from the frozen paper protocol.

The selected route and configuration are written before test `Y_test` is read.
After that write, test outcome is opened exactly once for scoring. Failure of
the approval gate must produce the direct fallback and is a valid result.

## Equal-budget competitors

The confirmatory comparison uses the same frozen row manifest and seeds 42–46.
Each tunable family receives exactly 30 validation candidates:

- plain MLP,
- FT-Transformer,
- V-REx,
- GroupDRO,
- monotone NN,
- Engression,
- linear-tail RBF,
- SVGP,
- PP-X.

If a package is unavailable or a method cannot train on the frozen hardware,
the row is marked `not run`; it is not replaced by a smaller historical search.
Candidate count equality is reported separately from runtime and parameter
count equality.

## Primary outcomes

1. PP-X versus the strongest completed equal-budget comparator:
   physical-unit paired log RMSE ratio.
2. Pooled R² and physical-unit macro RMSE.
3. Route selected, fallback/approval status, and validation evidence.

Secondary outcomes are five-seed stability, coverage if calibration-unit count
allows it, and compute cost. Six test units are insufficient for a conventional
two-sided sign test at 0.05; uncertainty is therefore reported without claiming
standalone test-cohort significance.

## Success and failure

Prospective support requires all of:

- no information-contract violation,
- nonnegative pooled R²,
- positive mean physical-unit log RMSE ratio against the strongest completed
  equal-budget comparator,
- no catastrophic unit with PP-X RMSE ratio above 2.

Any other result is reported as partial or failed prospective evidence. The
method, split, candidate set, thresholds, or headline metric must not be changed
after test outcome reveal.
