# Axial-fan untouched PP 확증 결과

## 판정

**확증 실패.** `1P_8F`, `4P_1F`, `4P_8F`의 공식 RUL 값을 열기 전에 PP와 matched MLP의 5-seed 예측을 모두 저장했다. 세 설정 모두 pooled R²가 음수였고 PP는 MLP보다 낮았다.

| 설정 | PP ensemble R² | plain MLP R² | PP RMSE | MLP RMSE | 우세 |
|---|---:|---:|---:|---:|---|
| 1P_8F | -0.375 | -0.068 | 35.644 | **31.420** | MLP |
| 4P_1F | -1.780 | -1.349 | 45.411 | **41.748** | MLP |
| 4P_8F | -0.289 | -0.132 | 32.369 | **30.337** | MLP |
| macro R² | -0.814 | **-0.516** | — | — | MLP |
| 전체 300 fan pooled | -0.746 | **-0.455** | 38.212 | **34.883** | MLP |

paired fan bootstrap의 `MLP RMSE - PP RMSE`는 -3.329이고 95% CI는 **[-4.374, -2.250]**이었다. PP가 우연히 진 정도가 아니라 이 고정 구조가 일관되게 더 나빴다.

## 최초 프로토콜의 스키마 오류와 봉인

Mendeley 설명에는 13열이면서 두 operating-point 열과 여덟 health 열이라고 쓰여 있어 한 열이 설명되지 않았다. 실제 파일은 `unit, time, operating 3개, health 8개`이고 train-only RUL 열은 없었다. 첫 프로토콜은 이 불일치로 중단했다.

스키마 감사 과정에서 `1P_1F` RUL만 화면에 출력됐기 때문에 이 설정 전체를 개발용으로 격리했다. 나머지 세 RUL 파일은 해시만 검증하고 별도 디렉터리에 봉인했다. 수정 프로토콜 커밋 `1c32b60` 뒤 모델 예측 전체를 `predictions_frozen_before_labels.npz`로 저장하고 그 다음에 세 RUL을 처음 읽었다.

## 왜 실패했나

train fan의 전체 수명 평균은 설정별 약 200 cycle이고 범위는 약 100–299였다. test의 실제 남은 수명은 평균 48–55, 표준편차 27–30, 최댓값 103–104였다. 반면 PP 예측의 표준편차는 41–47이고 최댓값은 169–224까지 뻗었다. 실제 RUL과 PP의 상관은 0.54–0.73으로 방향 신호는 있었지만 **예측 분산과 상단 꼬리가 지나치게 컸다.**

원인은 모델 깊이보다 selection task mismatch다. validation은 완전수명 unit의 매 5번째 시점을 모두 평균한 반면 공식 test는 각 unit의 무작위 truncated endpoint 하나만 평가한다. 전체 궤적 MSE로 고른 affine/residual scale이 endpoint 분포에서 과분산을 만들었다. C-MAPSS에서 성공한 운전조건 정규화는 operating offset은 제거하지만, 어떤 시점에 잘린 fan인지에 맞는 lifetime-scale shrinkage까지 정하지 못한다.

## 논문에서의 위치

- 300개 독립 official test fan을 사용한 진짜 dataset-level untouched negative cohort다.
- 결과를 본 뒤 feature, 모델, epoch 또는 seed를 다시 고르지 않았다.
- strict convex-hull 밖 100%라고 확인한 과제가 아니므로 **unseen-fan truncated-future RUL transfer**라고 부른다.
- 후속 구조는 train unit 안에서 endpoint censoring을 모사하는 nested validation과 lifetime-scale uncertainty head가 필요하다. 이 아이디어를 axial-fan test에 맞춰 개발하면 결과는 posthoc이므로, 다음 새 cohort에서 다시 동결 검증해야 한다.

재현 코드: `experiments/axial_fan_untouched_confirmation.py`
결과: `results/axial_fan_three_config_confirmation_v1/results.json`
봉인 선행 예측: `results/axial_fan_three_config_confirmation_v1/predictions_frozen_before_labels.npz`
