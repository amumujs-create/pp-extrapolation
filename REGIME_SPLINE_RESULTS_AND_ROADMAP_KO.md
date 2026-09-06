# Regime-spline PP 개발 결과와 논문 빌드업

## 모델

기존 PP의 frozen affine path 뒤에 degradation 축의 train 분위수 knot와 context-conditioned ReLU hinge slope를 추가했다. Tanh residual이 support 밖에서 포화되는 문제를 줄이고, NN이 regime별 외삽 기울기를 학습하도록 한 박사논문 확장용 개발 모델이다.

## Deep-future 결과

| 데이터 | 기존 PP single | 일반 NN ensemble | 자유 regime-spline single | 자유 spline ensemble | 단조 spline ensemble |
|---|---:|---:|---:|---:|---:|
| HUST | -1.069±0.682 | **0.340** | -0.236±0.491 | -0.106 | -4.241 |
| Virkler | -4.576±2.271 | -0.688 | -4.979±7.189 | -2.398 | **-1.705** |

자유 regime-spline은 HUST에서 기존 PP를 크게 회복했지만 양의 R²에는 도달하지 못했다. Virkler에서는 한 seed가 `0.423`이었으나 다른 seed가 폭주했다. 단조 residual slope는 Virkler 분산을 줄였지만 HUST를 심하게 손상했다.

단조 RUL prior를 residual 계수 부호로 구현한 것이 문제다. 전체 출력이 degradation 방향에서 감소해야 한다는 조건과 affine 대비 residual이 항상 음수여야 한다는 조건은 다르다. 다음 모델에서는 residual 부호가 아니라 전체 출력 Jacobian에 soft penalty를 적용해야 한다.

## 저널용 현재 스토리

1. 문제: unseen-unit strict late-tail extrapolation.
2. 모델: affine extrapolation path + bounded NN residual인 PP.
3. 실행 조건: validation residual evidence 기반 applicability certificate.
4. 근거: HUST, Virkler, NASA/RWTH 및 MATR에서 Ridge 대비 성능과 seed 안정성.
5. 실패 경계: MICH/FEMTO residual collapse, seen-unit deep-future regime shift.
6. 주장 제한: 모든 RUL 미래예측이 아니라 transferable late-tail relation이 있는 unit-level 외삽.

현재 regime-spline 결과는 저널 주 모델에 넣지 않고 failure analysis 또는 future-work 한 절로 둔다. 음수 R² 모델을 본편 novelty로 합치면 PP의 명확한 주장이 약해진다.

## 박사논문 빌드업

### Chapter 1 — Selective PP

현재 저널 연구. Transferable affine tail, residual evidence certificate, uncertainty-aware abstention을 정립한다.

### Chapter 2 — Latent regime transition PP

Seen-unit deep-future를 대상으로 change point를 latent variable로 학습한다. Regime 수를 고정 공식으로 넣지 않고 sequence encoder가 transition posterior와 regime별 slope distribution을 출력한다.

### Chapter 3 — Counterfactual future regimes

관측 prefix에서 가능한 future regime을 여러 개 생성하고, monotonicity는 residual 부호가 아니라 전체 prediction derivative와 boundary hitting time에 적용한다. Regime uncertainty를 RUL interval로 전파한다.

### Chapter 4 — Cross-domain prior assembly

배터리 capacity, crack length, bearing vibration처럼 도메인별로 다른 degradation coordinate를 contract로 선언한다. 사용 가능한 prior만 조립하고 evidence가 없는 모듈은 certificate가 끈다.

## 다음 고급 모델 사양

- Causal sequence encoder: GRU/TCN 중 validation 안정성이 높은 소형 모델
- Latent change-point head: transition probability와 regime duration
- Mixture-of-slopes decoder: regime별 affine slope와 연속 transition
- 전체 출력 Jacobian monotonicity penalty
- Counterfactual prefix augmentation
- 단일 모델 seed consistency regularization
- Applicability certificate가 승인한 데이터에서만 실행

이 구조는 HUST deep-future에서 개발하되, 기존 test를 다시 최적화하는 수치는 개발 결과로만 표시한다. 구조 고정 후 새로운 complete run-to-failure cohort에서 평가해야 박사논문 확장 증거가 된다.
