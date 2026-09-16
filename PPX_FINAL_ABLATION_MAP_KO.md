# PP-X v4 Final Ablation Map

## A1 — Residual correction

**Comparison:** Prior-only vs Prior+Residual

- positive direction: 6/6 settings
- normalized mean unit-RMSE reduction: +0.512
- setting-bootstrap 95% CI: [0.316, 0.676]
- individually significant after within-contrast correction: 4/6
- observed harm: 0/6

**Claim:** A learned nonlinear correction is required; the prior alone is insufficient.

## A2 — Prior-family diagnostic

**Primary matched comparison:** Affine vs BQ with the same residual/executor.

| Executor | BQ wins |
|---|---:|
| Unbounded residual | 3/3 boundary cohorts |
| Bounded residual | 3/3 boundary cohorts |

Validation selected BQ in all three factorial boundary cohorts. Consequently,
v4 prior selection and BQ-fixed cannot be distinguished by performance on
these cohorts. Optional prior selection is a methodological safeguard, not a
demonstrated main performance contribution.

## A3 — Typed executor

**Comparison:** Prior+Residual core vs Full PP-X typed executor.

- positive/neutral/negative settings: 4/2/0
- normalized mean unit-RMSE reduction: +0.215
- setting-bootstrap 95% CI: [0.065, 0.377]
- individually significant improvements: 3/6

**Claim:** No optional executor should be globally enabled; typed executors add
value selectively without observed harm in the evaluated settings.

## A4 — Validation routing

**Comparison:** one globally fixed executor policy vs typed validation routing.

Best fixed policy (transport) versus router:

- mean pooled R²: 0.838 → 0.886
- worst-setting pooled R²: 0.468 → 0.751
- router wins/ties/losses: 2/4/0

Across all evaluated fixed policies, the router had no setting-level loss.

The nested minimum used by v4 is algebraically equivalent to a flat minimum
over the same admissible prior×executor menu:

\[
\arg\min_p\min_e L(p,e)=\arg\min_{(p,e)}L(p,e).
\]

Therefore hierarchy itself is not claimed to outperform flat validation
selection. The contribution is **typed admissibility plus validation-selected
routing**, while the hierarchy supplies organization and auditability.

## A5 — Prior-conditioning control

The fully matched six-setting comparison now uses the same nonlinear capacity,
split, seed, optimizer, budget, initialization replay, and checkpoint rule.
The pure Battery contrast uses `frozen_unbounded_pp`; bounded/dual executors are
excluded because they would change correction capacity.

- positive/negative settings: 4/2
- normalized mean effect: +0.129
- setting-bootstrap 95% CI: [-0.139, +0.395]
- significant prior benefit/harm after BH: 2/2
- benefit: Sunwoda, RWTH; harm: HUST, MATR-b2
- MICH positive but not BH-significant; N-CMAPSS pooled R² positive but unit CI crosses zero

**Claim:** prior conditioning is heterogeneous and is not a universal
performance improvement. It should be described as a setting-dependent
extrapolation reference, while the correction ablation supplies the consistent
architectural contribution.

## Main-paper allocation

| Evidence | Placement |
|---|---|
| A1 Residual correction | Main |
| A3 Typed executor | Main |
| A4 Fixed policy vs routing | Main |
| A2 Prior family | Appendix/diagnostic |
| A5 fully matched six-setting prior-conditioning | Main or compact supplement |

All reported comparisons are retrospective. They do not replace a frozen
prospective cohort confirmation.
