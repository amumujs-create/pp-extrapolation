# NASA health-v2: real-data prior ablation

This is exploratory reuse of the previously inspected NASA cells; not independent confirmation. Four folds, five seeds, four arms: 80 fits. Exact same 255 hull-out test rows. Every variant was fixed before this run and is reported.

| Model | Pooled R² mean ± seed SD | Delta vs PP |
|---|---:|---:|
| PP | 0.495104 ± 0.003786 | +0.000000 |
| direction | 0.495104 ± 0.003786 | +0.000000 |
| transport | 0.469558 ± 0.010039 | -0.025545 |
| both | 0.469328 ± 0.010168 | -0.025776 |

## Diagnosis

- Baseline predictions replayed the original NASA health-v2 PP predictions exactly (maximum absolute difference 0, verified separately against the original archive).
- Direction-only loss was zero throughout training on the declared grid. It added no constraint beyond what the model already satisfied there.
- Transport loss was nonzero during optimization; this is not a disconnected-loss bug. It reduced the selected models’ average transport discrepancy but lowered held-out accuracy for all five seeds.
- The average raw-output transport discrepancy (unweighted over 20 fold/seed fits) changed from 1.760 to 1.619 cycles. Lower consistency error is not the same as lower prediction error.
- On B0005, mean per-cell R² fell from 0.7614 to 0.6415 with transport. All five B0005 fits selected epoch zero under transport, versus positive epochs under PP. That explains a substantial part of the loss on this split.
- The combined arm had temporary direction violations during training, even though the selected models had none on the design grid. This is why final diagnostics alone were insufficient.
- This result rejects the usefulness of this fixed transport configuration on this split. It does not rule out all priors, prove that the physical law is wrong, or justify tuning against these test results. Weight 10 and tolerance 2 cycles were not exhaustively tuned.
- Priors are assumptions about a scalar health coordinate. They do not establish identified causal interventions; confidence is not learned.

## Next falsifiable hypothesis

Assess the applicability of a prior on inner, unit-disjoint pseudo-extrapolation folds built from training data, before applying it at the outer boundary. Compare that policy against no prior and always-on prior, including shuffled reliability scores. This would test whether reliability selection itself helps; it is not implemented or validated by these results.

## Reproduction

Use `experiments/nasa_prior_ablation.py` and the fold archive preparation in `experiments/README.md`. `experiments/summarize_nasa_ablation.py` independently recomputes all 20 pooled scores from saved predictions. Source data are not bundled. Seeds quantify optimizer variation, not uncertainty over new battery populations.
