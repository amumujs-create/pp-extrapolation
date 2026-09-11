# Auto-Regime Weak-Prior PP-X v1.4 protocol

**Frozen:** 2026-09-10, 박진서. Stanford and ISU are already opened, so this
is retrospective architecture development only.

## Goal

Detect covariate regimes without test labels, choose a weak prior from a fixed
bank inside each regime, and return toward the baseline when the regime or
physical-unit evidence is unstable.

## Frozen structure

1. Fit three deterministic k-means regimes on robust-scaled **train x only**.
2. Assign validation/test rows with soft distance weights; no test-batch
   normalization or refitting.
3. Fixed prior bank: trust `.02,.05,.10,.20,.40`, each already averaged over
   four architectures and five seeds.
4. In each hard-assigned validation regime, run Stability-First group OOF with
   alpha restricted to `0,.05,.10,.15,.20,.25`; baseline weight is at least
   75%.
5. Regimes with fewer than five physical units use baseline only.
6. Blend regime routes by train-frozen soft weights so a regime transition is
   continuous.
7. Recheck assembled validation raw unit mean/CVaR/max regret at 2%/5%/10%.
   If needed, reduce every regime alpha by one common scale; if no positive
   scale is feasible, use the exact baseline.

## Comparators and reporting

Compare baseline, fixed RBPR, Stability-First v1.3, and auto-regime v1.4.
Report pooled/unit metrics, raw regret, accepted regimes, chosen trust/alpha,
regime occupancy, common safety scale, and exact fallback error.
Audit hard assignment and `K=2,4` as structural sensitivity; the primary
`K=3` choice must not be replaced after seeing test outcomes.

The regime detector identifies statistical operating states, not a new
physical law. Automatic physical prior invention is out of scope; the prior
bank must be declared before data outcomes are read.
