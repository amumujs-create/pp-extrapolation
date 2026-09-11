# Next untouched cohort — PP-X v1.2 continuous-portfolio protocol

**Model frozen before cohort selection:** 2026-09-10, 박진서.

The next cohort-specific protocol must freeze source, IDs, endpoint, eligibility,
unit split and strict-tail cutoff before opening trajectories. It must then use:

- architecture: width `{16,32}`, learning rate `{5e-4,1e-3}`, weight decay 2
- trust: `{0,.02,.05,.10,.20,.40}`
- seeds `42–46`
- baseline expert: fixed equal average of all four trust-zero
  width×learning-rate ensembles; no validation architecture winner
- one prior expert per positive trust: fixed equal average of all four
  width×learning-rate ensembles
- leave-one-physical-unit-out validation weight selection
- nonnegative convex weights summing to one
- group-balanced objective with worst-20% CVaR excess-loss penalty
- fixed regularization: `rho=.50`, baseline-priority `tau=.02`,
  prior-weight concentration `gamma=.01`
- final weights: coordinate-wise median of fold weights, renormalized

The portfolio weights are saved before test labels are read. Report the full
candidate grid, every fold weight, total prior mass, nominal weighted trust,
five seeds, pooled and unit-level metrics. Nominal weighted trust is only a
descriptor because each trust route is separately trained; it is not a literal
affine contribution fraction. Do not modify this policy after the first
new-cohort score.

The Stanford and ISU results are development evidence only. They cannot be
reported as confirmation of this frozen template.
