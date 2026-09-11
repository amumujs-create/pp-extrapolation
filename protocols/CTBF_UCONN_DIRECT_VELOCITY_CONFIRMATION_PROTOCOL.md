# Contract-normalized direct-velocity CTBF — UConn confirmation protocol

**Frozen before CTBF execution:** 2026-09-12, 박진서

## Status and scope

UConn–ISU–ILCC 1.2 Ah has previously been opened for PP-X, but no CTBF model
has been evaluated on it. This is therefore an external retrospective
model-concept confirmation, not a pristine prospective confirmation.

The hypothesis was selected after the Stanford/ISU Stage A result:
contract normalization plus a learned positive velocity field, without a
local-rate prior, improves boundary-tail RUL extrapolation.

## Data and split

Reuse the frozen eligibility, numeric-ID 60/20/20 physical-cell split, 25th
percentile train health cutoff, strict tail rows, and causal features from
`UCONN_ILCC_LOW_CAPACITY_PPX_V12_PROTOCOL.md`.

No cell, row, threshold, or split may change after CTBF execution begins.

## Contract normalization

UConn failure is 80% of each cell's median first-five capacity. Transform
health to

\[
z=(h-0.8)/(1-0.8),
\]

and transform all health means, standard deviations, and 1/3/5-RPT slopes by
the same affine scale. The declared failure boundary is then exactly zero.

## Confirmatory model

\[
\widehat T(z,c)
=
\int_0^z \frac{du}{v_\theta(u,c)},
\qquad
v_\theta(u,c)=\exp g_\theta(u,c)>0.
\]

There is no rate quotient, weak-rate anchor, prior ensemble, or
multi-timescale gate in the confirmatory model.

## Search and final fit

- widths: 16, 32
- learning rate: 0.001
- weight decay: 0.1
- quadrature points: 24
- maximum epochs: 350
- patience: 60
- selection seeds: 42, 43, 44
- final ensemble seeds: 42, 43, 44, 45, 46

Width is selected by validation pooled RMSE. The direct MLP receives the same
transformed inputs, width grid, optimizer budget, selection seeds, and final
seeds.

## Outcomes

- pooled R²/RMSE/MAE
- unit-macro R²
- per-cell RMSE win fraction
- 20,000-draw physical-cell bootstrap RMSE difference
- boundary maximum absolute prediction
- contextual comparison with the stored PP-X v1.2 result

## Success rule

Confirmation succeeds only if all hold:

1. CTBF pooled R² is positive.
2. CTBF pooled RMSE is lower than matched direct MLP.
3. At least 60% of test cells improve.
4. Physical-cell bootstrap 95% CI lower bound for
   `direct MLP RMSE - CTBF RMSE` is positive.
5. Boundary maximum absolute prediction is zero.

Stored PP-X is contextual because its optimizer grid differs; beating it is
not required for this confirmation.
