# Temporal prior assembly: first NASA and regime-signal audit

Exploratory development on the same four previously inspected NASA health-v2 cells.
All 255 test rows are identical to the original protocol and are outside the train
health-coordinate hull. Unit separation is preserved. The historical scalar PP
score is contextual: the temporal methods have additional past observations.
Four fitted arms x five seeds x four folds = 80 fits; prior-only is deterministic.

| Model | Pooled R² mean | Seed SD |
|---|---:|---:|
| Previous scalar PP (context only) | 0.495104 | 0.003786 |
| PP with current history statistics | -0.074458 | 0.001894 |
| GRU direct RUL | -1.088098 | 0.631363 |
| Online prior only | -0.804325 | — |
| GRU correction, fixed online gate | -0.343313 | 0.102593 |
| GRU correction and learned gate | -0.087151 | 0.071939 |

The learned gate and correction improve on prior-only, GRU-direct, and fixed-gate
in this configuration. They do not improve on scalar PP, and do not improve on
PP with history statistics either. This is a failed accuracy-improvement experiment.
The prior-based model should not replace the baseline. No test-based tuning was
performed within this run, but the design itself follows earlier inspected results.

## What can and cannot be inferred

The current code constructs two trend-to-boundary priors from signed short/long
OLS rates. This does not establish valid physics for every cell. Capacity recovery,
noise, and changing trends can make local slopes poor remaining-life predictors;
these are plausible explanations, not isolated causal findings from this experiment.
There are only two training cells in each fold. More temporal features may increase
estimation difficulty. The 25%-of-train-cap bounded residual can also limit repair
when both priors fail. Dedicated ablations would be required to separate these causes.

## Known-change synthetic audit

Ten fixed seeds for each of stable, abrupt-change, and gradual-change signals.
The short/long windows are 8/24. The alarm threshold is 0.35, sustained for 3
observations, after warmup. A true change starts at observation 75.

| Scenario | Runs with post-75 alarm | Median delay | Prechange alarm observations |
|---|---:|---:|---:|
| Stable | 0/10 | — | 0 |
| Abrupt | 10/10 | 6 observations | 0 |
| Gradual | 0/10 | Not detected | 0 |

These controlled traces support sensitivity to this abrupt change only. The signal
missed this gradual change and is not a calibrated regime probability. Real NASA
regime labels were not available, so detection accuracy on NASA was not measured.
No future unobserved regime can be inferred from this statistic alone.

## Next architectural requirements

1. A prior contract must express validity and uncertainty, not just boundary/direction.
2. A latent health estimator should separate reversible observation changes from
   irreversible degradation before forming rates. That estimator is not implemented.
3. Multiple priors can all fail simultaneously. A correction bound and normalized
   expert weights do not supply an abstention option or a calibrated RUL interval.
4. RUL should eventually be evaluated under explicitly stated future operation
   scenarios when load/temperature regimes can change; historical state alone does
   not specify future use. The present prototype assumes persistence of current trends.

These are research requirements, not verified enhancements. See
`TEMPORAL_PRIOR_DESIGN_KO.md` for implementation and reproduction commands.
