# Cross-fitted monotone RBPR development protocol

**Frozen:** 2026-09-10, 박진서. Stanford and ISU are already opened and are
used only for retrospective method development.

## Change from the first RBPR

Replace the fixed `1/(1+d+u)` modulation with

`m(x) = exp(-k_d * d(x) - k_u * u(x))`.

`k_d,k_u` are nonnegative, so prior mass cannot increase with distance or seed
disagreement. Candidate values are fixed at `0,.25,.5,1,2,4`.

For each positive trust and each held-out validation physical unit:

1. choose `k_d,k_u` and the largest feasible alpha on the remaining units;
2. require mean and worst-20% CVaR excess MSE to be within 2% of baseline;
3. predict the held-out unit once.

Choose trust by cross-fitted unit-mean MSE, subject to cross-fitted 2% risk
constraints. Aggregate fold parameters by coordinate median. If the aggregated
parameters violate full-validation constraints, reduce alpha only. Test labels
are read after this decision is frozen.

## Comparisons

Baseline, original fixed RBPR, cross-fitted global budget (`k_d=k_u=0`),
distance-only, disagreement-only, and full monotone budget. No test-selected
operating point may be adopted.
