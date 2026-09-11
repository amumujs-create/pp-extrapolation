# Hierarchical-risk RBPR development protocol

**Frozen:** 2026-09-10, 박진서. This is retrospective development on already
opened Stanford and ISU cohorts, not confirmation.

## Motivation and fixed estimator

ISU validation has many physical units with only one to three strict-tail rows.
Raw unit MSE therefore gives a one-row unit the same CVaR influence as a
well-observed unit. For each alpha, calculate paired unit excess MSE `e_g` and

`e_g^H = w_g e_g + (1-w_g) mean_g(e_g)`,

where `w_g = n_g/(n_g+5)`. The prior is fixed at `n0=5` before this run.
The mean-risk constraint continues to use unshrunk, equally weighted unit
excess. Only the worst-20% CVaR ranking and value use `e_g^H`.

## Selection

Use the same leave-one-physical-unit-out monotone RBPR procedure and 2% budget:

- trust `.02,.05,.10,.20,.40`;
- nonnegative distance/disagreement decays `0,.25,.5,1,2,4`;
- alpha grid `0,.01,...,1`;
- choose by cross-fitted unit-mean MSE subject to cross-fitted hierarchical
  CVaR and raw mean excess constraints;
- exact baseline fallback if no candidate is feasible.

The primary arm is full monotone RBPR with `n0=5`. Report raw CVaR alongside
hierarchical CVaR. Ablate `n0=0,2,10` and global/distance/disagreement terms,
but do not replace the primary arm based on test performance.
