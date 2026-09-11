# CTBF main contract benchmark and promotion audit

**Frozen before execution:** 2026-09-12, 박진서

## Purpose

Evaluate the contract-normalized direct-velocity CTBF on every PP-X main
setting for which a scalar ordered state, an outcome-independent failure
boundary, and a time-to-boundary target are all defensible.

## Included main settings

1. Sunwoda: discharge capacity to the declared 880 mAh crossing
2. RWTH: relative capacity to the declared 0.8 crossing
3. NASA PCoE: cell-normalized capacity margin to the 1.4 Ah crossing
4. Virkler: remaining crack-length margin to the observed 49.8 mm endpoint

Virkler is transformed from increasing crack length to decreasing remaining
margin before model fitting.

## Excluded settings

- HUST: target is a terminal-slope proxy because most cells do not reach the
  declared 0.880 Ah boundary.
- MICH: target is a published life label, not a common health crossing.
- MATR 2019 and MATR batch 2: end-of-record target and no declared boundary.
- N-CMAPSS: no scalar ordered state or declared failure boundary.

These exclusions are contract decisions, not performance-based exclusions.

## Coordinate adapters

- Sunwoda: `z = health_mAh - 880`
- RWTH: `z = health_ratio - 0.8`
- NASA: existing `z = (capacity - 1.4)/(initial_capacity - 1.4)`
- Virkler: `z = 49.8 - crack_length_mm`

All coordinates are positive before failure and exactly zero at the declared
boundary. Observed degradation rates, where present, are expressed as
negative `dz/dt`. Context features retain the existing PP-X benchmark rows and
splits.

## Primary CTBF

\[
\widehat T(z,c)
=
\int_0^z \frac{du}{v_\theta(u,c)},
\qquad
v_\theta(u,c)=\exp g_\theta(u,c)>0.
\]

Observed rate is used only as a weak local training loss where an aligned
causal rate exists; it is not a multiplicative prediction prior. NASA has no
rate feature and receives RUL supervision only.

## Exact search budget

Thirty validation candidates:

- width: `{16, 32, 64}`
- learning rate: `{2e-4, 5e-4, 1e-3, 2e-3, 5e-3}`
- weight decay: `{0.1, 2.0}`

Other settings:

- search seed: 42
- selected candidate refit seeds: 42–46
- maximum epochs: 350
- patience: 60
- quadrature points: 24
- physical-unit-balanced loss

NASA performs selection and refit separately inside each frozen
leave-one-cell-out fold before concatenating test predictions.

## Structural ablation

For the validation-selected configuration, refit five seeds for:

1. primary CTBF with local velocity supervision where available
2. CTBF without local velocity supervision
3. observed-rate boundary quotient where available
4. existing 30-candidate direct MLP
5. frozen PP-X
6. strongest existing 30-candidate comparator

The no-rate arm isolates the time-to-boundary architecture from local
transition supervision. NASA primary and no-rate are identical by design.

## Statistics and stability

- pooled R²/RMSE/MAE
- unit-macro R²
- physical-unit RMSE win fraction
- 20,000-draw unit bootstrap of paired log-RMSE ratio
- exact unit sign-flip test
- worst-unit RMSE ratio
- dataset-level exact sign test
- Benjamini-Hochberg correction across included datasets
- boundary maximum absolute prediction
- five-seed dispersion

## Promotion rules

CTBF may replace PP-X as the paper's universal main model only if:

1. it beats PP-X on at least 3/4 included settings;
2. mean dataset effect favors CTBF and at least 3/4 dataset effects are
   positive (the two-sided exact sign-test minimum with only four eligible
   settings is 0.125 and is reported descriptively, not used as an impossible
   \(p<0.05\) requirement);
3. no dataset has more than 10% pooled RMSE harm versus PP-X;
4. at least two datasets have unit-bootstrap CI entirely favoring CTBF;
5. boundary prediction is exactly zero everywhere;
6. performance does not depend entirely on the local-rate loss.

If these fail but CTBF has validation-detectable wins, it remains an optional
contract-specific executor. No excluded setting is counted as a loss.
