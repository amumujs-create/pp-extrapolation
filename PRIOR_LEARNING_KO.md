# PP의 prior 및 경계 변화율 전달 학습: 실험 버전

기본 PP에 선택적으로 더하는 학습 방법이다. 네트워크 구조는 유지하며 손실을 확장한다. 독창성과 실제 데이터의 성능 개선은 아직 검증되지 않았다. 과거 HUST/Virkler/NASA 점수는 이 버전의 성능이 아니다.

## 관계 prior

물리식을 RUL 출력 함수로 삽입하는 대신 사용자가 정의한 두 상태 사이의 예측 차이를 제한한다.

    lower <= f(changed) - f(anchor) <= upper

- 열화 개입 후 수명이 증가하지 않는다는 가정: [-inf, 0]
- 무관한 조건을 바꿔도 예측이 거의 같다는 가정: [-epsilon, epsilon]
- 알려진 변화량 범위: [lower, upper]

lower/upper는 원래 target 단위다. 각 관계의 confidence는 [0,1]이며 사용자 또는 학습자료만으로 결정한다. confidence=0이면 해당 관계는 손실에 기여하지 않는다. 신뢰도를 자유롭게 학습시키면 모두 0으로 만들어 손실을 회피할 수 있으므로 현재 버전에서는 고정한다.

    L_relation = mean(confidence * (relu(lower-delta)^2 + relu(delta-upper)^2))

실제 최적화는 target scale로 차이와 경계를 나눈 뒤 수행한다.

## 경계 변화율 전달

같은 타당한 개입 경로 위에 inner -> boundary -> outer 세 점을 정의한다. inner와 boundary는 train support에서 정하고 outer는 미관측 방향으로 이동한다. 외부 test 표본이나 test label은 사용하지 않는다.

    d_in = stop_gradient(f(boundary) - f(inner))
    d_out = f(outer) - f(boundary)
    r = outward_step / inward_step
    L_transport = mean(confidence * relu(abs(d_out - r*d_in) - tolerance)^2)

이는 관측 영역에서 학습한 국소 변화량을 외삽 영역에서도 일정 오차 내에서 유지하자는 가정이다. slope를 별도 물리식으로 지정하지 않는다. tolerance는 원래 target 단위다. 현재 구현은 사용자 제공 개입 경로와 신뢰도를 사용하며, 경로나 인과 그래프를 자동 발견하지 않는다.

## 전체 목적과 선택

    L = L_unit_balanced_MSE + lambda_relation * L_relation + lambda_transport * L_transport

prior 항은 clipping 전 네트워크 출력에 적용한다. affine 경로는 기존처럼 고정된다. prior는 soft penalty이므로 단조성이나 정확도의 전역 보장이 없다. frozen affine이 prior와 충돌하면 bounded correction이 모든 영역에서 해결할 수 없다. 내부 secant도 틀리거나 평평할 수 있으며 stop-gradient만으로 이를 방지하지 못한다. checkpoint는 기존과 같은 validation MSE로 선택하므로 prior를 가장 잘 만족하는 epoch가 선택된다는 보장도 없다.

각 prior의 타당성, lambda, tolerance, 개입 범위를 test를 열기 전에 고정해야 한다. 단일 센서만 바꾸면 다른 센서 및 과거 궤적과 모순될 수 있다. 여러 입력을 함께 갱신하는 feasible intervention을 사용해야 한다. 그런 구조적 가정 없이 생성한 표본은 인과적으로 식별된 반사실이 아니라 가상 perturbation이다.

## 사용 예

```python
import numpy as np
from pp_extrapolation import PriorPairs, fit_pp, predict

# changed_x는 train에서 생성한, 도메인상 타당한 열화 개입 상태.
n = len(train['x'])
prior = PriorPairs(
    anchor=train['x'], changed=changed_x,
    lower=np.full(n, -np.inf), upper=np.zeros(n),
    confidence=np.ones(n),
)
fit = fit_pp(train, validation, seed=42,
             prior_pairs=prior, prior_weight=1.0)
y_pred = predict(fit, test['x'])
```

`TransportTriples`는 `inner, boundary, outer, ratio, tolerance, confidence` 배열을 받는다. `fit_pp(..., transport_triples=triples, transport_weight=1.0)`으로 사용한다. 모든 배열은 원래 feature/target 단위이고 내부에서 train 통계로 변환한다. 학습 중 prior 표본은 별도 RNG로 minibatch sampling한다. 두 weight의 기본값은 0이며 기존 PP 실행을 재현한다. 현재 CSV CLI는 기본 PP이고 prior는 Python API로 전달한다.

## 재현 가능한 개발 실험

```bash
python examples/prior_ablation.py
```

동일 unit-disjoint 외삽 분할에서 PP, 방향 prior, transport, 둘 다, 잘못된 방향 prior를 비교한다. 부드러운 단조 함수와 외삽 중 방향이 바뀌는 함수를 포함한다. 둘 다 synthetic 개발 실험이며 real-data 일반화나 인과 추론 근거가 아니다. 점수는 seed별 pooled R²의 평균±표준편차다. 전체 결과는 `results/prior_ablation/results.json`에 저장된다.

## 기존 연구와 논문 기여 후보

단조성 정규화와 반사실 불변성 학습은 기존 연구가 있다.

- [Monotonicity Regularization, UAI 2022](https://proceedings.mlr.press/v180/monteiro22a.html)
- [Constrained Monotonic Neural Networks, ICML 2023](https://proceedings.mlr.press/v202/runje23a.html)
- [Neural Networks for Learning Counterfactual G-Invariances, ICLR 2021](https://iclr.cc/virtual/2021/poster/3239)

기여 후보는 신뢰도와 허용구간을 가진 관계 prior를 경계 변화율 전달과 결합하고, 외삽 거리·prior 오염도에 따른 이득과 실패를 같은 프로토콜로 규명하는 것이다. 짧은 선행연구 검색만으로 새로운 방법이라고 주장하지 않는다. 새로운 실데이터, weight/transport ablation, 잘못된 prior 및 다변수 개입 실험이 필요하다.

기존 설명의 주의점: bounded tanh correction은 raw network의 함수 형태를 제한하지만 예측 정확도를 보장하지 않는다. 배포 출력은 train RUL 상한으로 clip되므로 raw affine-tail 점근성과 같지 않다. 입력 feature 한 좌표에서 train 범위 밖이면 그 좌표를 포함한 전체 입력 hull에서도 밖임이 수학적으로 따른다. 다만 한 방향 밖이라는 사실은 여러 조건이 동시에 변하는 외삽을 검증한 것이 아니다.
