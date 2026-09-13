# Successor to PP-X: conjunctive acceptance criteria

Status: acceptance policy recorded during this audit; historical endpoints remain development data.
The NASA screen's specific rejection rule and configuration were saved by its
runner before training. This broader document is not a prospective registration.
User requirement: modeling novelty, coverage, and performance better than both PP-X and Engression.

## Interpretation

Prototype experiments are permitted to test a hypothesis. Promotion, expanded
development, or a success claim requires ALL gates below. A failed gate is a
rejection, not permission to delete settings, change the target, or tune against
the evaluation outcomes. This protocol does not guarantee that a superior model
can be found.

## 1. Modeling contribution

Specify a structural or learning assumption absent from both existing PP-X and
the closest primary literature. Provide a matched ablation showing its benefit.
Naming a combination of GRU, latent ODE, decoder, horizon loss and numerical
semigroup regularization is insufficient. In particular, semigroup agreement
of the SAME autonomous flow mainly tests its numerical solver and may improve
by collapsing dynamics to zero. Do not interpret it alone as external accuracy.

## 2. Coverage

- Applicability: retain every prespecified original setting; report unavailable
  inputs, unsupported tasks, and invalid outputs against the full denominator.
- Prediction coverage: every required test row receives a finite prediction;
  rejection/subset selection cannot inflate performance.
- Positive-performance coverage: report the count of settings AND independent
  cohorts with R2 > 0; require no decrease versus either comparator. A positive
  R2 count is not a confidence-interval coverage claim.
- Preserve both original support extrapolation (health, regime, operating
  covariates) and any temporal extension. Future prediction on another unit
  alone does not establish covariate extrapolation.
- Initial inventory: historical nine settings from
  `experiments/main9_anchored_residual_audit.py`, plus the Axial 3 settings,
  MATWI and Misata external-development settings. Axial's three variants are
  one cohort; other related datasets must also be grouped explicitly.

## 3. Matched performance

- Full frozen PP-X policy and official Engression must receive identical
  permitted input information, target definitions, unit splits, evaluation
  rows and validation rows. Preserve IDs and horizon coordinates in predictions.
- Match tuning trial allowance, seeds, ensemble size and clipping; report native
  optimizer/scaling differences, parameters, updates, wall time and sampling.
  An equal epoch cap is not a claim of equal FLOPs.
- Require strict RMSE improvement versus BOTH comparators on every prespecified
  setting. Report equal-cohort log RMSE ratios, unit-level uncertainty, worst
  unit risk, individual seed results and the equal-size seed ensemble.
- Before a superiority claim, require the upper confidence bound of the
  equal-cohort log RMSE ratio below zero for both comparisons. With too few
  independent cohorts, explicitly withhold that claim rather than bootstrap
  overlapping rows as independent observations.
- Full PP-X cannot be replaced by a weaker PP core to establish promotion.
  Failure against a core is useful for rejecting a candidate, not ranking the
  complete historical methods on a new target.

## 4. Confirmation and stopping

Use already-opened data for development only. A frozen new cohort evaluation is
required for prospective validity. Once a screen fails, retain all failures and
stop that candidate; a later changed architecture needs a separately recorded
hypothesis and remains post-test development on this screen.

## Current ETO audit

Novelty gate: not established. Latent ODE time-series extrapolation is already
explicit in [Rubanova et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf).
The initial ETO has no invariant-state factorization, operator certificate or
adjacent-prefix learning despite those being proposed in its plan.

Coverage gate: not established. The current API requires an ordered history,
future horizon and trajectory target. It cannot directly replace scalar PP-X
across the original operating-condition and health-support tasks.

Performance gate: the NASA screen tests a NEW measured-capacity target, with
train/validation horizons {1,2,4} and test horizons {8,16,32}. This demonstrates
query-horizon support exclusion (the horizon exceeds every train value), but
does not replicate the original PP-X RUL experiments. No promotion is possible
from that screen alone, even if all its comparisons favor ETO.

The previous synthetic numbers are withdrawn as superiority evidence: horizons
were shared by training and testing; `y[end+h]` was paired with a prefix ending
at index `end-1` (actual horizon h+1). The sanity tests only established code
execution, not scientific novelty or broad extrapolation performance.
