# 후속 연구 축: TSSM — Trajectory Survival State Model

> 현재 상태: 사용자 외삽 주제와의 불일치로 후속 후보에서 제외.
> 아래는 초기 제안 기록이다. survival prefix 사이의 등식은 추가 관측에 의한
> posterior 갱신을 무시하면 일반적으로 성립하지 않으므로 필수 제약으로
> 채택하지 않는다. 해당 손실은 구현되지 않았다.

## 한 문장

PP-X가 **구조화 prior를 안전하게 외삽 회귀에 사용**하는 방법이라면, TSSM은 센서 prefix에서 잠재 손상 상태를 필터링하고 그 상태의 **고장시간 분포**를 직접 추론하는 확률적 상태공간 모델이다. PP-X의 affine tail, residual, support gate, route selector는 TSSM에 넣지 않는다.

## 왜 별도 축인가

| 구분 | PP-X | TSSM |
|---|---|---|
| 예측 객체 | RUL 점 회귀 | 생존곡선 `S(tau | history)`와 분위수/RMST |
| 중심 가정 | contract로 prior를 허용한 뒤 residual을 제어 | 미래 손상은 감소하지 않고 event hazard는 양수 |
| 학습 신호 | 회귀 손실 + source route 검증 | event/censoring likelihood + prefix 간 시간 일관성 |
| 불확실성 | route/지원 범위 안전성 | failure probability, survival quantile, censoring 처리 |
| 배포 | 승인된 예측기 하나 | 같은 잠재 상태에서 horizon별 위험을 질의 |

따라서 “PP-X에 GRU나 hazard head를 하나 더 붙인 변형”으로 주장하지 않는다. 두 방법은 경쟁 baseline으로 비교하고, 어떤 경우에 서로 다른 실패 양상을 보이는지 보고한다.

## 모델

각 unit의 t 시점 관측 prefix를 `H_t={(x_i, time_i): i<=t}`라 한다. causal encoder가 context `c_t`와 현재 손상 `d_t >= 0`를 만든다.

```text
H_t -- causal GRU/filter --> (c_t, d_t)
                              |
future dynamics: d(s+ds)=d(s)+softplus(f_theta(c_t,d(s))) ds
                              |
hazard: lambda(s)=softplus(g_theta(c_t,d(s)))
                              |
S(tau|H_t)=exp[- integral_0^tau lambda(s) ds]
```

`d`는 구조적으로 비감소한다. hazard의 단조성은 강제하지 않는다. 초기 결함, 운전조건, 회복 효과가 있을 때 hazard가 damage 하나의 단조 함수라는 가정을 넣지 않기 위해서다. `c_t`는 관측 prefix에서만 계산되고, 미래 행·test batch 통계·test label은 사용하지 않는다.

현재 구현은 `src/pp_extrapolation/trajectory_survival.py`에 있다. 정규 격자 Euler 적분을 사용한 **v0 연구 골격**이며, 아직 SDE/CDE 또는 amortized posterior가 아니다.

## 학습 목표

관측된 failure row의 남은 시간 `T`는 다음 negative log likelihood를 쓴다.

`L_event = H(T|H_t) - log lambda(T|H_t)`,  `H(T)=integral_0^T lambda(s)ds`.

아직 failure가 나지 않은 unit은 임의 RUL label을 만들지 않고, censoring time `C`에 대해 `L_censor = H(C|H_t)`만 쓴다. train loss는 physical-unit 균등 가중이다.

논문용 v1에서는 여기에 아래 두 항을 반드시 추가한다.

1. **Prefix semigroup consistency.** 같은 unit에서 `t_a<t_b`인 두 prefix에 대하여, `S_a(delta_ab+u) = S_a(delta_ab) S_b(u)`가 되도록 log-survival 오차를 최소화한다. 이는 horizon별 RUL 회귀 head가 아닌 하나의 dynamics가 여러 prefix를 설명하는지 검증한다.
2. **Masked future-observation likelihood.** prefix 후의 센서 block을 복원/예측하는 auxiliary emission head를 훈련 중에만 사용한다. failure-time label이 드문 경우에도 상태가 관측 신호를 담게 하되, 배포는 survival head만 쓴다.

v0에는 event/censor likelihood만 구현되어 있다. semigroup과 emission은 ablation 가능하게 v1에서 넣어야 하며, 성능이 아닌 likelihood calibration과 prospective test로 채택 여부를 판정한다.

## 검증 가능한 주장과 금지할 주장

주장 가능 후보는 “censored partial trajectories를 포함해 coherent survival curve를 학습하고, 동일 unit의 prefix 간 conditional survival consistency를 측정한다”이다. NODE, latent degradation, RUL survival 자체는 선행연구가 있으므로 새롭다고 말할 수 없다.

특히 Neural ODE 기반 RUL은 이미 보고되어 있고, dynamic survival을 CDE latent state로 푸는 연구도 있다. TSSM의 차별점은 **고장 임계 prior를 쓰지 않고**, prefix-semigroup을 explicit evaluation/training object로 두며, RUL point error와 survival calibration을 같은 protocol에서 평가하는 설계에 한정한다. 이 역시 체계적 문헌조사와 실험 전에는 novelty claim이 아니라 가설이다.

## 실험 계획 (frozen before endpoint opening)

1. **입력 contract.** 각 row는 `(x: n×L×p, time: n×L, mask, groups, rul, event)`이다. `event=False`는 right-censored이고 `rul`은 censoring horizon이다. unit split을 먼저 수행한 뒤 prefix를 만든다.
2. **동일 정보 baseline.** GRU-direct RUL, temporal prior model, event-flow, DeepHit/DeepSurv류 survival baseline, TSSM-v0, TSSM-v1. PP-X는 static-feature 조건의 별도 reference로만 둔다.
3. **Primary endpoints.** unit-balanced time-dependent Brier score와 integrated Brier score, C-index, observed-failure RMST MAE, 80/90% survival interval coverage, 그리고 prefix semigroup error. RMSE만 primary로 두지 않는다.
4. **Ablation.** (a) monotone damage 제거, (b) censor loss 제거, (c) semigroup 제거, (d) emission 제거, (e) causal-prefix violation control. 각 ablation은 같은 parameter/seed budget으로 한다.
5. **Stress.** irregular sampling, withheld suffix length, new operating regime, and unfinished/censored units. Test labels 및 test-batch statistics로 threshold·route·calibration을 바꾸지 않는다.
6. **Success bar.** 미사용 unit cohort에서 baseline 대비 IBrier와 RMST MAE가 모두 악화되지 않고, 90% coverage가 사전 지정 tolerance 안에 있으며, semigroup error가 GRU-direct의 post-hoc survival conversion보다 작아야 한다. 하나라도 실패하면 기여 주장 대신 negative result로 기록한다.

## 다음 구현 순서

1. 현재 v0의 synthetic censoring/unit tests와 prefix leakage test를 추가한다.
2. pair sampler와 semigroup loss를 구현한다.
3. emission decoder 및 irregular-time solver를 추가한다.
4. sealed cohort protocol을 만들고, 그 뒤에만 NASA/배터리 external benchmark를 연다.

## 문헌 포지셔닝 (초기 스캔)

- McKee, *Remaining Useful Life Estimation Using Neural Ordinary Differential Equations* (IJPHM, 2021): NODE 기반 RUL은 이미 존재한다.
- Moreno et al., *Dynamical Survival Analysis with Controlled Latent States* (2024): controlled latent state로 dynamic survival을 다룬다.
- Zhong et al., *Physics-consistent liquid state-space network for RUL prediction* (Information Fusion, 2026): 단조 latent state와 RUL readout도 독립 노벨티가 될 수 없음을 보여 준다.

이 때문에 TSSM은 “first monotone latent RUL model” 같은 주장을 하지 않는다.
