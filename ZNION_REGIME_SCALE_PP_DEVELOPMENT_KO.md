# Regime-scale PP 개발 결과

Untouched Zn-ion 실패를 본 뒤의 **후속 개발 분석**이다. 독립 확증 성과로 쓰지 않는다.

## 구조

`regime-conditioned PP`는 두 prior path를 하나의 executor 안에 둔다.

1. prefix slope가 음수인 일반 열화 regime: 기존 boundary-quotient path
2. prefix slope가 양수인 activation/지연 열화 regime: 6개 prefix descriptor로 log-EOL을 학습하는 neural lifetime-scale head
3. `sigmoid(prefix_slope / 1e-4)`로 두 path를 관측치별로 연결

이는 서로 다른 완성 모델의 예측을 사후에 평균한 것이 아니라, PP 내부에서 현재 prefix가 나타내는 regime에 따라 RUL 생성 prior를 바꾸는 구조다.

## 결과

| 모델 | pooled R² | unit-macro R² |
|---|---:|---:|
| 동결 BQ-PP | -0.379 | -0.004 |
| plain MLP | -0.204 | -0.236 |
| regime-scale PP, seed mean | 0.489 | 0.554 |
| regime-scale PP, seed median | **0.741** | **0.726** |

seed-mean 기준 셀별 R²는 `435-1` 0.800, `438-3` 0.307로 두 regime에서 모두 양수로 회복했다. 중앙값 집계는 각각 0.801, 0.652다.

## 남은 문제

단일 seed pooled R²가 -1.803에서 0.885까지 변했다. 장수명 regime 학습 unit이 적어 neural scale head가 초기화에 민감하다. 따라서 다음 단계는 ensemble 성능을 최종 해법으로 채택하는 것이 아니라, nested out-of-fold lifetime target과 seed-consistency loss로 단일 head의 분산을 낮추는 것이다. 그 뒤 이 구조를 다른 untouched cohort에 다시 동결해야 한다.
