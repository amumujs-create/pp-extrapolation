# Boundary-Quotient PP 노벨티 감사

감사일: 2026-09-07

## 최종 판정

**BQ-PP는 단독 아이디어 최초성은 약하지만, PP 논문의 배터리 특화
모듈로는 중간 수준의 모델링 신규성이 있다.** 논문에서 주장할 대상은
`health margin × NN`이 아니라 다음의 전체 결합이다.

> Unit-disjoint strict health-tail RUL에서 RUL/health-margin quotient를 예측하되,
> 강하게 정규화한 affine tail을 먼저 적합하고 동결한 뒤 bounded causal-history
> residual만 학습하는 boundary-consistent PP.

이 조합과 동일한 RUL 모델은 이번 검색에서 확인하지 못했다. 다만 이는
법적·절대적 최초성 증명이 아니며, systematic review와 심사 과정에서 달라질 수
있다.

## 구성 요소별 선행연구 중복

| BQ-PP 요소 | 가장 가까운 선행연구 | 단독 노벨티 |
|---|---|---|
| `margin × NN`으로 EOL에서 정확히 0 | distance function을 NN에 곱해 경계조건을 정확히 만족하는 hard-constrained PINN | 약함 |
| 장애 threshold까지의 margin으로 RUL 계산 | first-passage/Wiener degradation model | 약함 |
| 이력으로 변화하는 degradation rate 학습 | adaptive drift, neural-driven stochastic degradation | 약함 |
| 구조 모델 + NN residual | structured-effect RUL NN, hybrid physics/data prognostics | 약함 |
| RUL/margin quotient에서 **frozen affine + bounded residual** | 정확히 같은 RUL 구성은 발견하지 못함 | 중간 후보 |
| 위 구조의 unit-disjoint, 100% late-health hull-out 다중 cohort 평가 | 대부분 ordinary holdout 또는 trajectory forecasting | 중간–강함 실험 기여 |

### 가장 가까운 문헌

1. Sukumar–Srivastava는 `distance-to-boundary × NN`으로 Dirichlet 경계를 정확히
   만족하는 방법을 제안했다. BQ-PP의 exact-zero gate와 수학적 패턴이 같으므로
   경계 곱셈 자체는 최초라고 하면 안 된다.
2. Kraus–Feuerriegel의 structured-effect RUL network은 population baseline, linear
   covariate effect, recurrent sensor-history component로 RUL을 분해했다. 따라서
   `해석 가능한 기본 경로 + NN`도 이미 있다.
3. Yang의 adaptive degradation process는 LSTM-CNN으로 Wiener drift를 학습하고
   Bayesian update를 수행한다. Long et al.도 Seq2Seq로 degradation-state-dependent
   drift를 학습했다. `NN이 변화하는 속도를 학습한다`는 최초가 아니다.
4. 2-phase Wiener RUL 연구는 change point와 unit heterogeneity까지 모델링한다.
   BQ-PP는 확률적 first-passage 분포를 제공하지 않으므로, 이 부분에서 더
   진보한 방법이라고 주장할 수 없다.

## 실험적 노벨티 근거

### 1. 기존 PP 대비 개선

| dataset | 기존 PP pooled R² | BQ-PP pooled R² | 차이 |
|---|---:|---:|---:|
| Sunwoda | 0.862 | 0.939 | +0.077 |
| RWTH | 0.506 | 0.878 | +0.372 |
| MICH | -1.522 | 0.468 | +1.990 |

같은 seed의 3 dataset × 5 seed = 15개 쌍에서 BQ-PP가 모두 개선했다.
단방향 Wilcoxon signed-rank `p=3.05×10⁻⁵`이다. 다만 BQ-PP는 세 dataset을
공동 학습했고 기존 PP는 dataset별 학습이므로, 이 검정은 전체 system 개선의
근거이지 frozen affine 단일 요소의 인과적 검정은 아니다.

### 2. frozen affine matched ablation

| arm | Sunwoda | RWTH | MICH | dataset-mean pooled R² | dataset-macro unit R² |
|---|---:|---:|---:|---:|---:|
| **frozen affine quotient + bounded residual** | **0.939** | **0.878** | **0.468** | **0.762** | **0.684** |
| trainable affine quotient + bounded residual | 0.900 | 0.855 | 0.319 | 0.691 | 0.491 |

나머지 설정을 같게 두고 affine만 학습 가능하게 만들면 세 dataset 모두
하락했다. frozen arm은 25 physical units 중 20개에서 RMSE가 낮았다.
평균 상대 RMSE 차이는 `-5.42%`이지만 unit bootstrap 95% CI는
`[-25.4%, +19.0%]`, Wilcoxon `p=0.071`이다. 방향은 일관되지만 현재 unit 수로
동결의 우월성이 통계적으로 확증됐다고 하기에는 부족하다.

### 3. 필수 구조 ablation

- affine quotient only는 Sunwoda `0.340`, RWTH `0.575`, MICH `-3.036`이다.
- bounded NN residual을 추가한 BQ-PP는 `0.939`, `0.878`, `0.468`이다.
- 따라서 성능은 hard boundary나 affine만으로 설명되지 않고, 학습된 nonlinear
  quotient residual이 필요하다.

## PAE와의 중복 감사

PAE의 `boundary_gated_shared_nn`도 `margin × positive NN`을 사용한다. 따라서
BQ-PP와 PAE 양쪽에서 exact-zero boundary architecture를 동시에 독립 핵심 노벨티로
주장하면 self-overlap 문제가 생긴다.

권장 분리는 다음과 같다.

- **PAE**: 관측 역할에서 사용 가능한 prior를 컴파일하고 prior-off를 선택하는
  typed prior system이 핵심이다. boundary NN은 executor 예시로 둔다.
- **PP**: frozen quotient tail, bounded residual, strict-tail geometry, validation transport와
  failure certificate가 핵심이다. BQ-PP는 boundary가 관측될 때만 켜는 선택 모듈로
  둔다.

PAE를 먼저 게재한다면 PP는 PAE를 자기 인용하고, boundary multiplication이 아닌
**frozen quotient residual이 strict extrapolation에서 주는 효과**만 추가 기여로 주장해야
한다.

## 투고 가능한 노벨티 문장

> We introduce a boundary-quotient prior-residual parameterization for strict-tail
> RUL extrapolation. Unlike direct RUL regression or a fully trainable hard-boundary
> network, it freezes a regularized affine quotient tail and restricts the history
> network to a bounded correction, separating extrapolative trend from cohort-specific
> nonlinear deviation.

사용하면 안 되는 문장은 다음과 같다.

- the first boundary-constrained neural RUL model
- the first hybrid physics–NN RUL method
- the first model to handle changing health–RUL relationships
- universally robust to unseen degradation mechanisms

## 상위 저널용으로 남은 검증

1. **Matched hard-boundary controls**: 같은 encoder·parameter budget의 direct NN,
   soft boundary-loss NN, fully trainable hard-boundary NN, frozen BQ-PP 비교.
2. **더 많은 unit의 frozen-path 통계**: 현재 25-unit CI가 0을 포함한다.
3. **Leave-one-cohort-out**: 새 cohort에서 zero-label, one-unit calibration, full supervised
   경우를 분리해 relationship-shift 범위를 검증한다.
4. **확률 비교**: two-phase Wiener/FPT와 point accuracy뿐 아니라 uncertainty calibration,
   NLL/coverage를 함께 비교한다.
5. **고정 후 새 cohort 확증**: 현재 BQ-PP는 이미 본 세 dataset에서 개발됐다.

## 핵심 선행연구

- Sukumar & Srivastava, *Exact imposition of boundary conditions with distance
  functions in physics-informed deep neural networks*, CMAME 2022,
  https://doi.org/10.1016/j.cma.2021.114333
- Kraus & Feuerriegel, *Forecasting remaining useful life: Interpretable deep
  learning approach via variational Bayesian inferences*, DSS 2019,
  https://doi.org/10.1016/j.dss.2019.113100
- Yang, *Adaptive Degradation Process with Deep Learning-Driven Trajectory*,
  https://arxiv.org/abs/2103.11598
- Long et al., *A neural-driven stochastic degradation model for state-of-health
  estimation of lithium-ion battery*, JES 2024,
  https://doi.org/10.1016/j.est.2023.110248
- *Remaining Useful Life Prediction for Two-Phase Hybrid Deteriorating Lithium-Ion
  Batteries Using Wiener Process*, IEEE Access 2024,
  https://doi.org/10.1109/ACCESS.2024.3374776
