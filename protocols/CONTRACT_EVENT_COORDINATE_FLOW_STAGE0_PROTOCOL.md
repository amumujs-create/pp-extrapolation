# Contract-conditioned event-coordinate flow — 9-setting Stage 0

**Frozen before execution:** 2026-09-12, 박진서

## Objective

Test a single model family that has PP-X-level dataset coverage and a genuine
forward-level structural change. The model represents RUL as an inverse
positive-velocity integral over an event coordinate. The coordinate is
observed when a defensible physical boundary exists and learned from causal
state otherwise.

## Model

For context representation \(c_\phi(x)\), event margin \(m_\eta(x)>0\), and
positive velocity \(v_\theta>0\),

\[
\widehat T(x)
=
\int_0^{m_\eta(x)}
\frac{du}{v_\theta(u,c_\phi(x))}.
\]

Two contract modes share the encoder, velocity operator, quadrature, and loss:

1. **Observed event coordinate:** \(m_\eta(x)\) is replaced by the declared
   physical remaining margin.
2. **Latent event coordinate:** \(m_\eta(x)\) is a learned positive scalar.
   A within-unit temporal-order loss requires the margin not to increase as
   the event approaches. A gauge loss fixes its otherwise arbitrary scale.

This is not a soft mixture with PP-X and has no PP-X fallback in Stage 0.

## Settings and modes

- observed: Sunwoda, RWTH, NASA, Virkler
- latent: HUST, MICH, MATR 2019, MATR batch 2, N-CMAPSS

Reuse the exact train/validation/test rows and groups from the nine-setting
equal-budget benchmark. NASA retains its four frozen folds.

## Stage-0 budget

This is a falsification screen, not the final equal-budget result.

- widths: 16, 32
- learning rates: 0.0005, 0.001
- weight decay: 0.1
- candidates per setting: 4
- search seeds: 42, 43, 44
- selected ensemble: same three seeds
- maximum epochs: 300
- patience: 50
- quadrature points: 24
- group-balanced RUL loss

## Ablations

1. full event-coordinate flow
2. latent mode without temporal-order loss
3. direct MLP reference from the stored 30-candidate benchmark
4. frozen PP-X reference

Observed-coordinate settings do not have a latent-order ablation.

## Outcomes

- coverage: finite predictions for all 9 settings
- pooled R²/RMSE and unit-macro R²
- unit win fraction versus PP-X and direct MLP
- worst-unit RMSE ratio
- boundary exactness for observed-coordinate settings
- latent margin temporal violation rate on train and validation
- seed dispersion

## Stage-0 pass rule

Proceed to a 30-candidate five-seed benchmark only if:

1. finite predictions are produced for 9/9 settings;
2. pooled R² is positive on at least 7/9 settings;
3. event-flow beats the 30-candidate direct MLP on at least 6/9 settings;
4. it beats PP-X on at least 3/9 settings;
5. no setting has more than 50% pooled RMSE harm versus PP-X;
6. latent-order loss improves at least two latent settings and materially
   harms no more than one.

Failure is retained as a rejected unified-model challenger. Stage-0 results
must not replace the paper's final equal-budget evidence.
