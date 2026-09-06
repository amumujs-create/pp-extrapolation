# Total-output Jacobian PP: deep-future 개발 결과

## Jacobian의 의미

이 연구에서 Jacobian은 예측 RUL이 입력에 얼마나 민감한지를 나타내는 미분이다. 열화 진행 좌표를 `q`라 하면 핵심 값은 `d y_hat / d q`이다. 정상적인 RUL 궤적은 열화가 진행될수록 감소해야 하므로 양의 값은 국소적인 물리 위반이다.

기존 monotone spline은 NN residual의 기울기만 음수로 제한했다. affine 경로의 오차를 residual이 위쪽으로 보정해야 할 때도 이를 막아 성능이 크게 나빠졌다. 새 방법은 affine+spline+local residual의 **전체 출력** Jacobian에만 다음 soft penalty를 적용한다.

`L = L_data + lambda * mean(ReLU(d y_hat / d q)^2)`

또한 train 관측점에서만 미분을 제한하면 미관측 미래에서 다시 기울기가 뒤집혔다. 그래서 train context의 라벨은 복사하지 않고, train-validation 간 진행 폭으로 미래 `q`를 생성한 counterfactual ray에서 penalty를 계산한다. test label과 test feature는 사용하지 않는다. `lambda`는 validation MSE로 seed별 선택한다.

## 결과 (pooled R²)

| 데이터/방법 | single-seed 평균 ± SD | 5-seed prediction ensemble |
|---|---:|---:|
| HUST free regime-spline | -0.236 ± 0.491 | -0.106 |
| HUST Jacobian-ray regime-spline | **0.117 ± 0.353** | **0.255** |
| Virkler free regime-spline | -4.979 ± 7.189 | -2.398 |
| Virkler Jacobian-ray regime-spline | -4.979 ± 7.189 | -2.398 |

HUST에서는 평균, 분산, ensemble R²가 모두 개선됐다. 특히 이전 free spline의 음수 ensemble을 양수로 바꿨다. 그러나 plain NN ensemble 0.340에는 소폭 못 미친다. Virkler에서는 validation이 모든 seed에서 `lambda=0`을 선택해 모델이 원래 free spline으로 되돌아갔다. 즉 제약이 해로운 경우 사용하지 않는 안전장치는 작동했지만, Virkler의 regime 전환 문제는 해결하지 못했다.

## 해석과 논문상 위치

이 결과는 Jacobian 제약 자체보다 **미관측 미래 ray에서 전체 예측의 방향성을 정규화한다**는 설계가 중요하다는 증거다. HUST처럼 열화 방향이 비교적 연속적인 데이터에는 유효하다. Virkler처럼 sparse crack-length 단계와 급격한 성장률 전환이 있는 경우, 단일 연속 기울기 prior는 부족하며 latent change-point 또는 regime mixture가 필요하다.

현재 결과는 이미 본 HUST/Virkler에서 수행한 post-hoc 개발 결과다. 저널의 확증 결과로 주장하려면 구조와 `(lambda, ray horizon)` 선택 규칙을 고정한 뒤 untouched deep-future cohort에서 검증해야 한다. 현 단계에서는 박사논문의 다음 모델 장으로 쓸 수 있는 유망한 중간 결과이며, 기존 PP의 unseen-unit late-tail 주 결과를 대체하지 않는다.
