# Support-adaptive PP 개발 결과

## 방법

Soft-anchor PP에 관측별 외삽 gate를 추가했다. train-only 표준화 공간에서 각
특징의 train 범위를 벗어난 거리를

`d(x) = ||relu(z_min-z) + relu(z-z_max)||_2`

로 계산하고 NN 보정을 다음처럼 감쇠한다.

`y_hat = f_affine(x) + exp(-beta*d(x))*r_NN(x)`

train support 안에서는 gate가 1이라 NN이 데이터를 학습한다. support 밖으로
멀어질수록 불안정한 NN 보정은 줄고 외삽 가능한 affine prior tail이 남는다.
`beta in {0,.1,.3,1}`와 soft-anchor 강도는 validation MSE로만 선택했다. test
label이나 test hull 거리는 선택에 사용하지 않았다.

현재 거리는 계산이 안정적인 axis-aligned support 거리이며 정확한 다차원
convex-hull Euclidean 거리는 아니다. 논문에서는 이 둘의 ablation이 필요하다.

## Strict tail / unseen-unit pooled R²

Seeds 42–46 평균±표준편차다.

| 데이터 | Frozen PP | Soft-anchor PP | Support-adaptive PP | matched plain NN |
|---|---:|---:|---:|---:|
| HUST | 0.724±0.039 | 0.813±0.062 | **0.910±0.013** | 0.758±0.032 |
| Virkler | 0.857±0.019 | 0.869±0.030 | **0.886±0.025** | -0.604±0.472 |
| NASA health | 0.495±0.004 | 0.513±0.002 | **0.513±0.002** | 0.233±0.094 |

HUST는 frozen PP 대비 +0.186, matched plain NN 대비 +0.152다. Virkler도
frozen PP 대비 +0.030이고 NASA의 soft-anchor 개선은 유지됐다. 세 데이터 모두
validation 선택 후 평균 성능이 악화되지 않았다.

## 논문 해석 제한

이 실험은 이미 본 세 데이터에서 수행한 retrospective 방법 개발이다. 새로운
데이터셋이나 잠근 cohort에서 같은 선택 규칙을 변경 없이 실행해야 prospective
근거가 된다. 또한 일반적인 OOD residual shrinkage, 고정 beta, Mahalanobis 거리,
정확한 convex-hull 거리와 비교해야 support-conditioned prior trust의 독립 기여를
판별할 수 있다.

원시 결과: `results/support_adaptive_pp/{hust,virkler,nasa}/results.json`
