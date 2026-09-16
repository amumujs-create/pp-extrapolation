# PP-X Ablation Matching Audit

## 결론

- **Residual contribution (Prior-only → Prior+Residual):** 6-setting nested comparison으로 유효.
- **Typed executor contribution (Prior+Residual → Full PP-X):** 6-setting nested comparison으로 유효.
- **Prior conditioning (Direct NN → Prior+Residual):** 현재 6-setting 결과는 전부 capacity-matched가 아님. heterogeneous diagnostic으로만 사용.
- **Prior type (Affine vs BQ):** Battery 3 cohort에서 동일 residual/executor를 맞춘 factorial comparison으로 유효.

## Matching status

| Setting group | Direct control | Prior+Residual | Matching judgement |
|---|---|---|---|
| Sunwoda/RWTH/MICH | width 64 direct NN, seeds 42–46, shared optimizer/budget family | width 64 BQ residual, seeds 42–46 | matched control available |
| HUST | stored GroupDRO competitor | affine-prior PP core | not fully capacity matched |
| MATR-b2 | stored GroupDRO competitor | affine-prior PP core | not fully capacity matched |
| N-CMAPSS | stored GroupDRO competitor | latent-regime PP | not fully capacity matched |

Battery matched-control metadata:

```text
features=11, width=64, learning_rate=1e-3, weight_decay=0.01,
seeds=[42,43,44,45,46]
```

Matched Battery-3 result (`bq_pp` vs `direct_nn`):

- 25 physical units
- mean relative RMSE delta: **-0.333** (BQ PP lower)
- unit bootstrap 95% CI: **[-0.539, -0.118]**
- paired Wilcoxon two-sided p: **0.00278**

## Paper wording guardrail

Allowed:

> Prior conditioning showed heterogeneous behavior in the six-setting direct-control diagnostic. A strictly capacity-matched control was available for the three boundary cohorts, where BQ PP improved aggregate physical-unit error relative to the matched direct network.

Not allowed:

> All six Direct-NN versus Prior+Residual comparisons were capacity matched.

## Final ablation roles

1. **A1 Residual correction:** Prior-only vs Prior+Residual.
2. **A2 Prior-family diagnostic:** Affine vs BQ under the same residual/executor.
3. **A3 Typed executor:** basic residual vs typed executor.
4. **A4 Validation routing:** global fixed executor vs typed validation routing.
5. **A5 Prior-conditioning control:** Battery-3 matched result as primary matched evidence; six-setting result as heterogeneous supporting diagnostic.
