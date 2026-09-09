# Axial-fan PP 실패 복구 개발 결과

## 결론

최초 untouched 실패는 데이터 열이나 RUL 정렬 오류가 아니라 **학습·검증 시점
분포와 공식 endpoint 시험 분포의 불일치**가 주원인이었다. 공식 test label을 본
뒤 수행한 개발 실험에서 마지막 105-step 위험구간으로 학습 범위를 맞추고,
affine prior와 neural residual의 기여도를 한 PP 네트워크 내부 gate가 정하도록
바꾸자 세 설정 모두 양의 R²와 matched MLP 대비 우위를 보였다.

| 설정 | 최초 PP R² | 수정 PP R² | 동일 조건 MLP R² | PP RMSE | MLP RMSE |
|---|---:|---:|---:|---:|---:|
| 1P_8F | -0.375 | **0.607** | 0.577 | **19.050** | 19.778 |
| 4P_1F | -1.780 | **0.532** | 0.302 | **18.624** | 22.755 |
| 4P_8F | -0.289 | **0.612** | 0.581 | **17.754** | 18.465 |
| 전체 300 fan pooled | -0.746 | **0.591** | 0.502 | **18.484** | 20.412 |

수정 PP의 macro R²는 0.584이고 MLP는 0.486이다. 전체 fan 단위 paired
bootstrap에서 `MLP RMSE - PP RMSE`는 1.931, 95% CI는
**[1.078, 2.834]**였다. 설정별 차이는 4P_1F에서 명확했고, 1P_8F와
4P_8F의 개별 95% CI는 0을 조금 포함하므로 개별 설정의 유의한 우월성까지
주장하지 않는다.

## 무엇을 고쳤나

1. 최초 학습은 약 270-step 전의 큰 RUL까지 포함했지만 공식 test는 fan당
   하나의 late endpoint만 평가했다. 마지막 105-step 위험집합으로 train과
   validation을 맞춰 예측 상단의 과분산을 제거했다.
2. frozen affine prior를 항상 전량 더하는 대신, 하나의 PP 안에서 학습된 gate가
   affine 경로와 residual 경로의 기여도를 관측별로 나눈다. 별도 완성 모델들의
   사후 ensemble은 아니다.
3. 폭 32, 학습률 5e-4, weight decay 2.0과 5개 seed를 세 설정에 공통으로
   사용했다. 각 seed의 epoch만 validation으로 선택했다.

## 증거의 범위

이 결과는 **성공적인 구조 복구 및 mechanism evidence**이지만 독립 외부 확증은
아니다. 105-step 위험구간과 gate 구조는 최초 공식 test 결과를 확인한 뒤 개발됐다.
따라서 논문에서는 최초 untouched 실패와 수정 개발 결과를 함께 제시하고, 같은
nested endpoint-matching 절차를 test label 없이 고정 적용한 다음 새 데이터셋을
확증 증거로 사용해야 한다.

재현 코드: `experiments/axial_fan_tail_gate_pp_development.py`

결과: `results/axial_fan_tail_gate_pp_development_v1/results.json`
