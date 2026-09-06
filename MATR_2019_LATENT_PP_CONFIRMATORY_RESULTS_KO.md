# MATR 2019 latent-transition PP 사전고정 확증 결과

## 판정

프로토콜을 데이터 다운로드 전에 공개 커밋 `ab103c8`로 고정한 뒤 MATR 2019-01-24 cohort를 한 번 평가했다. 사전 정의한 primary pooled 기준에서 **confirmatory success**였다. 그러나 unit-macro R²는 음수이므로 모든 cell에 균일하게 일반화됐다는 뜻은 아니다.

## 사전 적합성

| 항목 | 결과 |
|---|---:|
| 원자료 cell | 45 |
| 적합 cell | 44 |
| train / validation / test cell | 26 / 8 / 10 |
| train / validation / test-tail windows | 17,003 / 1,306 / 2,235 |
| validation train-hull 외부 비율 | 100% |
| test train-hull 외부 비율 | 100% |

Cell index 31은 `QDischarge`가 834개, `cycle`이 833개로 구조 길이가 달라 사전 규칙에 따라 제외했다. 용량이나 수명을 확인해 선택적으로 제외한 것이 아니다.

## 결과

| 모델 | ensemble pooled R² | pooled RMSE | unit-macro R² |
|---|---:|---:|---:|
| Ridge | -1.796 | 125.28 | -2.053 |
| Plain NN | -0.184 | 81.52 | -0.150 |
| Original PP | -0.666 | 96.69 | -1.052 |
| Jacobian-ray PP | -0.139 | 79.95 | -0.593 |
| **Latent-transition PP** | **0.257** | **64.56** | -0.511 |

| 모델 | single-seed pooled R² 평균 ± SD |
|---|---:|
| Plain NN | -0.477 ± 0.304 |
| Original PP | -0.670 ± 0.027 |
| Jacobian-ray PP | -0.164 ± 0.064 |
| **Latent-transition PP** | **0.171 ± 0.109** |

사전 기준 다섯 항목, 즉 latent ensemble이 plain NN과 PP를 능가하고, single-seed 평균도 두 모델을 능가하며, seed SD가 plain NN보다 크지 않아야 한다는 조건을 모두 통과했다.

## 해석

HUST에서 개발한 latent late-tail expert가 별도의 2019 battery cohort에서도 pooled 외삽 성능을 재현했다. 특히 단순 ensemble 최고점만이 아니라 단일 재학습 평균과 seed 안정성도 plain NN보다 좋았다. 따라서 `affine 안전 경로 + 학습형 regime gate + early/late neural experts`라는 구조에 데이터셋 밖 증거가 생겼다.

반면 test gate 평균은 seed별 0.997~1.000이었다. 이는 test-tail이 모두 late regime으로 배정됐음을 뜻하며, test 안에서 두 regime을 혼합했다는 뜻은 아니다. 또한 unit-macro R²가 음수여서 pooled 성공이 cell별 성공으로 이어지지 않았다. 논문에서는 primary pooled 성공과 이 이질성을 같이 제시하고, per-cell 실패를 applicability certificate의 다음 대상으로 삼아야 한다.

Jacobain-ray PP는 기존 PP보다 좋아졌지만 pooled R²가 여전히 음수였고 test 기울기 위반도 seed별 45.5~100%였다. 연속 방향 제약만으로는 cohort shift를 해결하지 못했고, late-regime expert의 표현력이 실제 개선의 핵심이었다.

## 재현성

- Protocol commit: `ab103c8`
- 평가 스크립트: `experiments/matr_2019_latent_confirmatory.py`
- 기계 판독 결과: `results/matr_2019_latent_confirmatory/results.json`
- 사전 test-source 감사: `results/matr_2019_latent_confirmatory/eligibility_pretest.json`

이 결과 이후 MATR 2019는 더 이상 untouched가 아니다. 추가 조정 결과는 모두 development로 분류해야 한다.
