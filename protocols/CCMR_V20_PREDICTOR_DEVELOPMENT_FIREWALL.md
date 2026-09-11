# CCMR v2.0 predictor-development firewall

Frozen before implementation and development evaluation: 2026-09-11,
박진서.

## Holdout exclusion

The following cohorts and every artifact derived from their features, labels,
predictions, errors, or competitor rankings are forbidden during architecture
selection, hyperparameter selection, ablation decisions, and stopping:

- Alloy A;
- Multi-Stage RPT.

They may be loaded only after the implementation, development-cohort results,
deployment thresholds, and this protocol have been hashed into the frozen v2.0
manifest. They are already-opened historical datasets, so the final operation
is an auditably separated holdout replay, not a new confirmatory experiment.

## Development evidence

Only other longitudinal trajectory cohorts may affect design:

- Luminosity condition shift;
- LED ordered tail;
- NASA discharge trajectory;
- Concrete HPC-to-UHPC;
- LG M50T Expt1;
- SIT LFP;
- RADAR NMC.

Unavailable or schema-incompatible cohorts are reported and not replaced by a
holdout cohort. Development selection uses leave-one-cohort-out summaries where
at least three compatible cohorts are available.

## Frozen design family

The v2.0 candidate is an anchor-residual causal dynamics expert bank:

1. zero correction (exact persistence);
2. robust linear multi-scale rate;
3. damped acceleration;
4. monotone spline residual;
5. small nonlinear residual.

Only present and past observations may enter an expert. Train physical-unit
cross-fit predictions determine sign consensus and dispersion. Validation
physical units determine nonnegative convex expert weights and deployment
mass under raw mean/CVaR20/maximum regret caps of 0%/1%/2%. Context-support and
causal shadow gates remain mandatory. Any failed certificate restores exact
persistence.

## Selection objective

Among risk-feasible candidates, minimize the equally weighted mean
validation-unit MSE. Ties choose the lower-complexity candidate and then the
smaller deployment mass. A candidate with no positive macro improvement over
persistence is rejected.

Development acceptance additionally requires:

- leave-one-cohort-out false-accept rate no greater than v1.9;
- aggregate raw maximum regret no greater than 2%;
- exact fallback replay error equal to zero.

No criterion may be changed after either holdout loader is invoked.
