# PP-X 다음 연구 축: ETO — Extrapolative Trajectory Operator

> 현재 상태: **채택하지 않음; 추가 ETO 구조 개발 중단.** NASA의 미학습 horizon
> 평가에서 Engression보다 평균·최장 horizon·최악 cell RMSE가 높았다.
> latent ODE 대비 독립 노벨티와 기존 PP-X 전체 적용 범위도 미입증이다.
> 아래 내용은 최초 가설이며 구현 완료/성공 주장이 아니다.
> 근거: `ETO_NASA_SCREEN_DECISION_KO.md` 및
> `protocols/SUCCESSOR_EXTRAPOLATION_ACCEPTANCE_PROTOCOL.md`.

## 정정된 연구 질문

이 축의 본체는 RUL/survival이 아니라 **외삽**이다.

> 관측 prefix만으로 식별한 동역학 연산자가, 보지 못한 미래 시간 구간과 다른 unit에서 실제 trajectory를 얼마나 정확하고 일관되게 계속 생성하는가?

RUL은 extrapolated health trajectory가 미리 선언된 failure boundary를 처음 통과하는 시간으로만 산출한다. 경계가 없으면 trajectory forecast만 평가한다. 따라서 PP-X처럼 `static x -> scalar y`의 affine-prior/residual 보정도 아니고, survival head를 따로 최적화하는 방법도 아니다.

## ETO 구조

```text
causal prefix H_t
  -> filter E_phi(H_t) = z_t
  -> latent flow dz/dtau = f_theta(z)       [source에서 하나만 학습]
  -> decoder D_psi(z_{t+tau}) = y_hat(t+tau)
```

`y_hat(t+tau)`는 오직 `z_t`를 `tau`만큼 적분한 뒤 decoder를 통과해서 나온다. 미래 horizon마다 독립적인 RUL/trajectory head를 두지 않는다. 그래서 멀리 갈수록 단기 패턴을 복사하는 회귀가 아니라, 학습된 flow가 어떤 오류를 누적하는지를 직접 검증할 수 있다.

초기 구현은 [src/pp_extrapolation/trajectory_operator.py](src/pp_extrapolation/trajectory_operator.py)이다. causal GRU filter + autonomous latent neural flow + decoder이며, zero-initialized flow로 persistence에서 출발한다. PP-X 코드나 route selector를 호출하지 않는다.

## 모델링 노벨티 후보

Neural ODE/CDE, latent dynamics, trajectory forecasting 각각은 선행연구가 있다. 따라서 아래 **결합된 extrapolation contract**만 기여 후보로 둔다.

1. **Suffix-only multi-horizon operator training:** 한 unit의 prefix를 여러 시점에서 자르고, 이후 실제 suffix 전체만 target으로 써서 하나의 flow를 학습한다. horizon 별 head를 두지 않는다.
2. **Flow-semigroup constraint and endpoint:** `Phi_(a+b)(z)`와 `Phi_b(Phi_a(z))`의 차이를 학습·보고한다. 이는 긴 horizon 외삽이 실제로 동일한 dynamics의 합성인지 시험한다.
3. **Unit-conditional, environment-invariant dynamics:** v1에서 `z=(z_mechanism,z_environment)`로 분리한다. flow는 `z_mechanism`에만 작용하고, operating condition은 encoder/decoder에는 들어가지만 미래 mechanism flow를 test batch에 맞춰 바꾸지 않는다. 환경이 바뀌면 예측값과 함께 *operator validity score*를 내고, score가 낮으면 trajectory를 생성하되 extrapolation claim을 하지 않는다.
4. **Forecast-then-functional evaluation:** RUL, failure probability, threshold crossing은 공통 forecasted trajectory에서 사후 계산한다. 이를 통해 같은 동역학이 trajectory·boundary time 모두에서 맞는지 판정한다.

이것은 “first neural ODE for RUL”이나 “first monotone state model” 주장이 아니다. NODE 기반 RUL과 controlled-latent dynamic survival은 이미 있다 ([McKee 2021](https://doi.org/10.36001/ijphm.2021.v12i2.2938), [Moreno et al. 2024](https://arxiv.org/abs/2401.17077)).

## 학습 데이터 형식과 누수 방지

각 train record는 `(prefix x[0:t], time[0:t], future horizons, future y, physical-unit ID)`로 만든다. `future y`는 반드시 그 prefix 뒤의 suffix이다. unit split을 먼저 하고, test unit의 suffix·batch statistic·endpoint로 encoder, flow, normalizer, threshold를 다시 고르지 않는다.

`MultiHorizonBatch`는 이 contract를 코드 수준에서 받는다. V0은 MSE masked suffix rollout이다. V1 loss:

`L = L_suffix + lambda_sg L_semigroup + lambda_adj L_adjacent-prefix + lambda_inv L_environment-invariance`.

- `L_suffix`: 1, 2, 4, 8, ... horizon의 held-out suffix forecast.
- `L_semigroup`: 동일 latent state의 flow composition 차이.
- `L_adjacent-prefix`: 실제로 한 observation이 더 들어온 prefix와, 먼저 forecast해 도달한 state의 decoder agreement.
- `L_environment-invariance`: source environment label이 있을 때 mechanism state에 예측 가능한 environment 정보가 남지 않도록 하는 adversarial loss. label 없이는 이 항을 쓰지 않는다.

## PP-X와 공정한 실험 설계

| 문제 | PP-X | ETO |
|---|---|---|
| primary target | single endpoint scalar | future trajectory 전체 |
| external extrapolation | support 밖 scalar mapping | withheld future suffix + unseen unit trajectory |
| training | contract-approved prior/residual | suffix-only latent flow |
| RUL | 직접 scalar 예측 가능 | extrapolated curve의 threshold crossing |
| 평가 | unit RMSE/R²와 route safety | horizon-wise RMSE, long-horizon slope error, semigroup error, boundary-crossing MAE |

공통 데이터에서는 관측 prefix 길이와 target horizon을 정확히 맞춘다. PP-X가 유리한 scalar endpoint만으로 ETO를 판정하지 않고, ETO가 유리한 full-trajectory metric만으로도 판정하지 않는다. primary는 horizon별 unit-balanced error area(AUC), 최장 horizon RMSE, physical-unit worst-case regret다. RUL이 있는 cohort에서는 crossing-time MAE를 추가한다.

## 다음 실행 순서

1. NASA/배터리 adapter에서 **unit split 후** multi-prefix/suffix batch builder를 구현한다.
2. persistence, GRU direct multi-horizon, NODE/CDE류 동일-budget baseline과 v0을 비교한다.
3. semigroup·adjacent-prefix loss와 invariant factorization을 하나씩 추가한다.
4. 새 sealed cohort에서 protocol을 먼저 고정한 뒤 최장 horizon 결과를 한 번만 연다.

성능이 아직 없으므로 ETO를 논문 방법으로 확정하거나 PP-X의 후속 성공이라고 주장하지 않는다. 현재는 외삽을 명시적으로 목표로 삼는 구현 가능한 연구 가설이다.
