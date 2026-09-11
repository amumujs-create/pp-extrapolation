# CTBF contract-normalized Stage A protocol

**Frozen before execution:** 2026-09-12, 박진서

## Question

Did the first CTBF challenger fail on ISU because it used a global normalized
boundary and a noisy one-step velocity prior? Stage A corrects those two
specifications without opening a new cohort.

## Data status

- Stanford and ISU 250 mAh are retrospective development cohorts.
- Physical-unit train/validation/test IDs and tail rows are unchanged from
  the PP-X v1.1 experiment.
- Test labels are used only once after validation selection.

## Contract-normalized coordinate

For cell \(i\), let \(q_{0,i}\) be the median initial capacity and
\(q_{f,i}\) the declared failure threshold. Define

\[
z_{it}
=
\frac{q_{it}/q_{0,i}-q_{f,i}/q_{0,i}}
{1-q_{f,i}/q_{0,i}}.
\]

This maps every declared failure boundary to \(z=0\). Health means, standard
deviations, and slopes are transformed by the same affine scale. Stanford has
\(q_{f,i}/q_{0,i}=0.8\). ISU uses the released fixed threshold
\(q_{f,i}=0.200\) Ah, hence its normalized boundary is cell-specific.

## CTBF arms

1. direct MLP on the transformed features
2. direct positive-velocity CTBF
3. lag-1 weak-rate CTBF
4. robust multi-timescale CTBF

For the multi-timescale arm, the weak velocity prior is the median of
floor-truncated degradation magnitudes:

\[
v_{\mathrm{weak}}
=
\operatorname{median}_{k\in\mathcal K}
\max(-\dot z_k,\epsilon).
\]

Stanford uses lag-1 and five-step window slope. ISU uses lag-1, lag-3, and
lag-4 slopes. The train-derived rate floor is fixed before validation/test
prediction.

## Search budget

- seeds: 42, 43, 44
- widths: 16, 32
- weak-prior log-residual bounds: 0.5, 1.0, 2.0
- learning rate: 0.001
- weight decay: 0.1
- maximum epochs: 350
- patience: 60
- quadrature points: 24
- physical-unit-balanced training loss

Configuration selection uses validation pooled RMSE; numerical ties select
the smaller width and smaller bound.

## Primary comparisons

- contract-normalized multi-timescale CTBF vs transformed direct MLP
- multi-timescale CTBF vs lag-1 CTBF
- contract-normalized CTBF vs the earlier global-boundary CTBF
- contextual comparison with stored PP-X v1.1 on the identical rows

Report pooled R²/RMSE, unit-macro R², per-unit win fraction, and 20,000-draw
physical-unit bootstrap RMSE differences.

## Stage A pass rule

Proceed to path-consistency Stage B only if all hold:

1. Stanford multi-timescale CTBF RMSE is no worse than 75 cycles.
2. ISU multi-timescale CTBF RMSE is below 2.50 cycles.
3. The multi-timescale arm beats lag-1 CTBF on at least one cohort and causes
   no more than 5% RMSE harm on the other.
4. Boundary prediction is exactly zero in both cohorts.

Promotion over PP-X requires a later frozen external confirmation and is not
decided in Stage A.
