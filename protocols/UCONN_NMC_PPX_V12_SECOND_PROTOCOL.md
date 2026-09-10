# UConn–ILCC NMC/Gr — second frozen PP-X v1.2 protocol

**Frozen:** 2026-09-10, 박진서. This file was written before downloading or
unpickling any NMC trajectory.

## Independent cohort

- Source: UConn–ILCC NMC/Gr Battery Aging Dataset (2025).
- Public metadata only: 44 Panasonic UR18650AA cells, 11 cycling conditions,
  ambient temperature, RPT approximately every 100 cycles, explicit EOL at
  65% of initial capacity.
- Compact source: authors' processed NMC per-cell pickle files in
  `REIL-UConn/rapid-soh-estimation-from-short-pulses`.
- This is chemically and experimentally independent of the prior 64-cell LFP
  attempt. No LFP threshold or result is reused.
- Workspace audit found no prior UConn NMC, Sanyo processed-file, or REIL cohort
  evaluation. The repository tree was inspected only for names and byte sizes.

## Eligibility and split

1. Use every numbered `processed_data/NMC/All/Sanyo *.pkl` cell containing at
   least 15 finite, ordered RPT capacity observations and an observed first
   crossing of 65% of the median first-five capacities.
2. Schema/readability exclusions are reported. No curve-shape, lifetime, group,
   or model-based removal or replacement is allowed.
3. Sort numeric cell IDs. First 60% train, next 20% validation, final 20% test.
4. Stop as inconclusive below 30 eligible cells, six validation cells, or six
   test cells.

## Strict tail and model

- Target: cycles remaining to the first observed 65% crossing.
- Causal channels: current normalized capacity, 1/3/5-RPT degradation slopes,
  causal five-RPT mean/std, and current cycle index. Cell identity and test
  batch statistics are prohibited.
- Train cutoff: 25th percentile of all eligible pre-event train health rows.
  Fit only rows strictly above the cutoff; define the realized boundary as the
  minimum retained health. Score held-out rows strictly below that boundary.
- Stop as infeasible if validation or test has fewer than 50 scored rows or
  held-out coverage below the boundary is not 100%.

Architecture and optimization are unchanged from the first v1.2 protocol:
width `{16,32}`, learning rate `{5e-4,1e-3}`, weight decay `2.0`, trust
`{0,.02,.05,.1,.2,.4}`, selection seeds `42–44`, final seeds `42–46`,
300 epochs and patience 50.

Positive trust is accepted only when validation pooled RMSE improves by at
least 2%, at least 60% of validation cells improve, worst-cell RMSE ratio is
at most 1.10, and the cell-bootstrap 95% lower bound of matched-MLP minus prior
RMSE is positive. Otherwise use exact trust-zero matched MLP. Trust-zero replay
error must be at most `1e-5`.

Validation R² at or below zero makes the test number `uncertified`. Success
requires test pooled R² above zero and RMSE no worse than matched MLP; positive
trust is additionally required to claim prior-expansion success.

No criterion may change after the first NMC trajectory is opened.
