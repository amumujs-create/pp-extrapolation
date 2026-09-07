# MATR2019 latent PP 실패 원인과 validation calibration

## 행·cell 단위 진단

기존 strict capacity-tail split과 저장된 동일예산 FT 예측을 행 단위로 정렬했다. Latent PP는
FT보다 pooled R²가 낮지만 예측과 정답의 상관은 더 높았다.

| 진단 | Latent PP | FT-Transformer |
|---|---:|---:|
| pooled R² | 0.257 | 0.331 |
| prediction-target correlation | **0.725** | 0.682 |
| 평균 RUL bias | -25.2 cycles | -17.2 cycles |
| unit-macro R² | -0.511 | -0.076 |

PP는 RUL 0--55와 173--315 구간에서 FT보다 RMSE가 낮고, 55--173의 중간 구간에서
크게 뒤졌다. 거리 중간 tertile에서도 PP가 우세했지만 가장 먼 tertile에서는 FT가
우세했다. Test 10 cells 중 PP가 5개, FT가 5개에서 낮은 RMSE를 보였다.

Train에 없던 충전 정책의 test cells `c38, c41, c42`에서는 모두 PP가 FT보다 낮은
RMSE를 냈다. 따라서 unseen policy 자체가 PP 패배의 주원인이라는 가설은 지지되지 않는다.

## 구조적 원인

Latent gate 평균은 0.9983이고 전체 test 행의 96.6%에서 0.99를 넘었다. 두 expert를
사용해야 하는 latent-transition 구조가 test tail에서 사실상 late expert 하나로 붕괴했다.
그럼에도 PP의 prediction-target correlation은 높았다. 즉 주요 오차는 순서와 곡선 형태보다
validation에서 test로 이동할 때 발생한 출력 scale과 offset의 miscalibration이다.

## Leave-one-validation-cell-out calibration

Test label을 사용하지 않고 validation cell을 하나씩 제외해 `identity`, global bias,
group-balanced bias, bounded affine calibration을 비교했다. 다섯 PP seed 모두 affine을
선택했다. 최종 calibrator는 모든 validation cell로 적합하고 test에는 한 번 적용했다.

\[
\hat y_{cal}=\operatorname{clip}(a\hat y_{PP}+b,0,y_{max}^{train})
\]

| 모델 | 보정 전 ensemble R² | 동일 보정 후 ensemble R² | 보정 후 unit-macro R² |
|---|---:|---:|---:|
| Latent PP | 0.257 | **0.466** | 0.012 |
| FT-Transformer | 0.331 | 0.377 | **0.031** |

Calibrated PP의 단일 seed R²는 `0.473, 0.457, 0.459, 0.299, 0.455`, 평균은
`0.429 ± 0.073`이다. 동일 규칙으로 보정한 FT의 단일 seed 평균은 `0.326 ± 0.030`이다.
따라서 PP의 개선을 미보정 FT와만 비교한 결과가 아니며, 공정한 보정 후에도 PP가 pooled
R²에서 0.088 높다.

## 판정과 한계

이 결과에서 PP의 MATR 성능 문제는 해결됐다. 최종 개발 모델은 `validation-calibrated
latent PP`이고 pooled R²는 0.466이다. 이는 모델 ensemble이나 FT와의 예측 혼합이 아니라
PP 단일 출력의 validation-only calibration이다.

다만 calibration은 MATR test를 이미 확인한 뒤 제안한 사후 개발 요소다. 기존 0.257의
사전 확증 기록을 대체할 수 없으며, 0.466은 개발 결과로 별도 표시해야 한다. 다른
데이터셋의 validation에서 identity fallback이 적절히 선택되는지 확인하고, 전체 규칙을
고정한 뒤 새로운 untouched cohort에서 재검증해야 일반적 개선으로 주장할 수 있다.

기계 판독 결과:

- `results/matr_pp_ft_failure_diagnostic_v1/results.json`
- `results/matr_pp_validation_calibration_v1/results.json`
- `results/matr_ft_validation_calibration_v1/results.json`

## 교차 데이터 확인

동일한 validation-only 규칙에 seed consensus fallback을 추가해 7개 데이터셋에 적용했다.
MATR, HUST, RWTH는 각각 `0.257→0.466`, `0.773→0.899`, `0.507→0.743`으로 개선됐고,
나머지 4개는 identity fallback으로 기존 점수를 유지했다. 자세한 결과는
`OUTPUT_CALIBRATION_CROSS_DATASET_KO.md`에 기록했다.
