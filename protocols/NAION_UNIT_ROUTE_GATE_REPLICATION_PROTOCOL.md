# Frozen unit-route gate replication on Na-ion group 5

This is a second prospective application of the already frozen v4 unit-route
gate. It is registered after group-4 scoring and before downloading or inspecting
group-5 files. No model, threshold, feature, or decision rule changes from
`NAION_UNIT_ROUTE_GATE_V4_PROTOCOL.md`.

- Cohort: `270040-5-[1-8]-*.csv`, lexicographic order.
- Split: first four train, next two validation, final two test.
- Eligibility, boundary, causal features, survival projection, five seeds, global
  validation checks, train-distance threshold 3.0, and validation-distance
  threshold 1.5 are unchanged.
- The gate decision and prefix commitment are persisted before test outcomes.
- Success requires at least one approved evaluable test unit, correct route for
  every evaluable unit, and positive selective PP pooled R2 on approved units.

