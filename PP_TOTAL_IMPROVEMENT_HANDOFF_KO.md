# PP 전체 개선 분석 및 후속 구현 인계서

작성일: 2026-09-08. 범위: PP 전체 방법론과 최근 Na/Zn-ion 확장. 초기 분석 이후 P0 scale 감사와 A0–A4 Zn-ion 개발 실험까지 실행했다. PAE 모델을 PP에 합치는 지시가 아니다.

## 0. 이번 실행에서 해결된 부분

- 현재 raw-cycle BQ residual의 이론적 보정 폭 중앙값은 0.125 cycle인데 affine 절대오차 중앙값은 80.771 cycle이었다. bound 안에서 필요한 오차가 도달 가능한 행은 0.228%였다.
- 단위를 정규화해 보정 폭을 넓인 후보는 단기 unit 4개 중 3개를 개선했지만, 장수명 unit을 제외한 fold에서 실패했다. nested aggregate pooled R²는 기존 raw BQ 0.115, normalized additive -0.175, multiplicative -6.354였다. support shrinkage도 inner validation이 beta=0을 골라 실패했다. 이 후보들을 최종 모델로 채택하지 않는다.
- 기존 BQ의 epoch/config 선택 후 validation prefix를 포함해 전체 development prefix로 고정 epoch refit하는 누락을 수정한 후보가 Zn-ion v3 replay에서 pooled R² **0.822→0.909**, unit-macro R² **0.375→0.690**으로 개선됐다. 세 test unit R²가 모두 양수였다.
- 동일 full-development refit MLP는 Zn-ion pooled R² 0.528이었다. PP의 0.909 우위가 동일 재학습 계약에서도 유지됐다.
- 같은 80%-EOL prior를 쓰는 Na-ion 관측 cohort에서도 BQ-PP가 no-refit 0.781에서 refit 0.824로 올랐다. 동일 refit MLP는 0.304였다. Na-ion PP의 unit-macro R²는 0.806이고 5개 test cell 모두 양의 R²였다.
- 공동 dual-scale battery executor의 기존 refit도 재실행해 Sunwoda/RWTH/MICH 0.934/0.842/0.751과 residual bound 위반 0건을 재현했다.
- 최종 refit의 seeds 42–46 pooled R² 범위는 0.909257535–0.909257578로 사실상 동일했다. 현재 residual bound가 매우 작아 seed 영향이 작은 것과 일관된다.

Zn-ion 0.909와 Na-ion 0.824는 이미 관측한 test에 대한 사후 개발 결과다. 새로운 untouched 확증 점수로 쓰지 않는다. 개선의 주원인은 새 NN 구조가 아니라 **동일하게 선택된 epoch으로 validation prefix supervision을 포함한 표준 full-development refit**이다. 같은 refit 계약을 비교 baseline에도 적용해야 한다.

## 1. 결론과 실행 우선순위

가장 먼저 할 일은 더 큰 네트워크가 아니라 **보정 폭의 단위 정합성**이다. 그다음 scale-aware residual, causal history, unit을 제외한 모듈 선택 순서로 진행한다. 복수 expert와 dynamic gate는 마지막 후보로 둔다.

| 우선순위 | 작업 | 해결하려는 문제 | 채택 전 필요한 증거 |
|---|---|---|---|
| P0 | 각 adapter의 target·margin·residual 단위 감사 | 같은 bound가 도메인별로 전혀 다른 의미 | 보정 가능 폭/필요 오차 비율, 실제 residual 기여 |
| P0 | 입력·라벨·split·refit 계약 통일 | 모델 효과와 supervision 차이가 섞임 | 동일 manifest와 같은 정보량 baseline |
| P1 | normalized BQ와 multiplicative scale residual 비교 | affine bias를 작은 가산 residual로 수정 불가 | unit holdout 개선 및 단위 변환 일관성 |
| P1 | affine fitting 목적과 ridge 크기 재설계 | quotient target 불안정, 표본 수에 따른 regularization 변화 | 중복행 및 경계 근처 민감도 검사 |
| P2 | masked multiscale history | 초기 짧은 이력·짧은 slope의 잡음 | history arm만 바꾼 paired 비교 |
| P2 | support·regime 증거를 분리한 residual 신뢰도 | support 거리와 prior 타당성의 혼동 | shell 및 unit별 실패 예측 능력 |
| P3 | OOF로 학습한 small dynamic gate | 고정 slope gate의 전이 대응 부족 | 단일 residual을 넘는 추가 이득 |
| P3 | error-risk head와 구간 | 낮은 seed disagreement 속 공통 bias | held-out risk calibration·구간 폭/coverage |

## 2. 기존 결과로 확정된 것과 미확정인 것

- 공동 dual-scale BQ는 Sunwoda/RWTH/MICH에서 0.934/0.842/0.751 ensemble pooled R²를 기록했다. 각 데이터의 개별 최고점을 동시에 넘은 것은 아니다. 고정 BQ의 Sunwoda/RWTH 0.939/0.878보다 낮지만 MICH를 복구한 tradeoff다.
- MATRb2 ablation에서는 transport가 큰 효과를 냈다. decay 단독 ensemble 개선은 약 0.002이고, transport와 결합했을 때 추가 효과가 있었다. 이를 decay 단독의 보편적인 정확도 이득으로 설명하지 않는다.
- 최근 Zn-ion 개발 PP 0.822는 RBF-only 0.780보다 높지만, 이 차이만으로 **NN residual의 기여**를 증명하지 못한다. BQ의 affine 경로만으로도 차이가 날 수 있다.
- Zn-ion MLP에 development EOL 정보를 맞춰 주자 -0.082에서 0.537로 상승했다. 해당 refit도 엄밀한 nested unit CV는 아니므로 공정 비교의 중간 결과다.
- 기존 최종 통합 표는 여러 executor와 서로 다른 프로토콜의 묶음이다. Zn-ion 한 가지 구조 개선을 전체 PP 개선으로 이름만 바꾸면 안 된다.

근거: `FINAL_PP_BENCHMARK_TABLE_KO.md`, `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md`, `UNIFIED_DUAL_SCALE_PP_IMPROVEMENT_KO.md`, `ZNION_MATCHED_INFORMATION_CONTROLS_KO.md`.

## 3. 새로 확인한 최우선 병목: Zn-ion residual의 절대 보정 폭

코드 경로:

- `experiments/znion_bq_confirmatory.py::make_rows`: y는 남은 cycle 수이며 time_scale로 나누지 않는다. time_scale은 feature에만 사용한다. margin은 SOH−0.8이다.
- `src/pp_extrapolation/boundary_quotient.py::components`: 출력은 `m × softplus(a+r)`, bounded correction은 `r=B×tanh(raw)`이다.
- 최근 Zn-ion 설정은 `B=0.5`이다.

softplus의 미분이 0과 1 사이이므로, 같은 affine 경로에 대해 정확하게

\[
|\hat y_{BQ}-\hat y_{affine}|\le m B
\]

가 성립한다. **m=0.2일 때 최대 보정은 0.1 cycle**이다. m이 그보다 큰 관측도 있을 수 있으므로 전체 최대치를 0.1이라고 단정하지 말고 실제 margin 배열로 계산한다. 수십~수백 cycle의 bias를 이 residual만으로 수정하기는 어렵다. 이는 실행으로 확인한 성능 개선이 아니라 구현에서 도출되는 표현력 상한이다.

반면 공동 배터리 adapter는 normalized target을 예측하고 time_scale을 다시 곱해 평가한다. 같은 B라도 cycle 단위 보정 폭이 달라진다. 이 문제를 모든 BQ 실험의 버그라고 일반화하지 말 것.

첫 진단 출력:

1. 각 adapter의 y 단위, time_scale, margin 정의, B, 실제 mB의 분포.
2. `abs(yhat_BQ-yhat_affine)`와 `abs(y-yhat_affine)`의 unit별 분포.
3. affine+BQ residual 대 affine-only를 같은 RBF 경로/gate에서 비교.
4. 선택 epoch, residual saturation 비율, gradient norm과 train/validation loss.

## 4. P1 후보: scale-aware BQ

### 후보 A: 최소 변경 normalized BQ

train에만 의존하는 양의 시간 척도 T를 정하고 `y*=y/T`로 학습한다.

\[
\hat y=T m\,softplus(a(x)+B\tanh r_\theta(h_t)).
\]

예측에 T를 한 번만 복원한다. EOL, margin, loss, ridge 초기화, checkpoint 선택까지 동일한 단위를 사용한다. bound는 normalized quotient 공간의 값이 된다. 기존 feature time_scale과 output time_scale을 명시적으로 구분한다.

### 후보 B: relative scale correction

\[
\hat y=T m\,softplus(a(x))\exp[b\tanh r_\theta(h_t)].
\]

이는 가산 0.1 cycle보다 큰 상대 보정을 허용하면서 frozen affine quotient를 유지한다. `exp(-b)≤yhat/yaffine≤exp(b)`는 yaffine>0에서 성립한다. m=0에서는 두 예측이 모두 0이다. 이 구조는 제안이며 성능·노벨티가 확인되지 않았다. affine quotient가 거의 0이면 곱셈 보정도 회복하지 못하므로 A와 반드시 비교한다.

권장 초기 후보: A의 B∈{0.5,2,6}, B의 b∈{0.5,1,2}. 이것은 사전 실행 예산 제안이며 최적값이 아니다. 먼저 동일 작은 encoder로 비교하고 gate를 동시에 추가하지 않는다.

### affine 초기화와 loss

현재 `_affine_initialization`은 `softplus_inverse(y/m)`에 ridge를 맞춘다. 작은 m에서 target이 커지고, 최종 RUL 오차 최소화와 달라질 수 있다. 다음을 하나씩 비교한다.

- 기존 inverse-quotient ridge.
- normalized RUL loss `loss(m softplus(a), y/T)`로 affine만 먼저 fitting 후 freeze.
- affine fitting은 두 번째 방식을 쓰되 Huber loss와 MSE를 inner validation에서 비교.

현재 ridge는 가중 SSE의 합에 alpha를 더한다. weight 합이 행 수에 따라 변하므로 같은 alpha의 상대 강도가 달라진다. 가중 손실을 weight 합으로 나눈 formulation과 alpha 변환을 명시한다. 기존 ridge를 무조건 버리지 말고, 모든 행을 동일 횟수 복제했을 때 새 formulation의 해가 보존되는지 확인한다.

주 평가 지표는 사용자 요구대로 **pooled R²**다. 학습 중 unit-balanced loss와 pooled loss를 비교할 수 있으나, 짧은 unit의 R²를 직접 loss로 쓰면 작은 분모에 지배될 수 있다. macro R²와 unit RMSE는 보조 안전 지표로 유지한다.

## 5. P2 후보: 이력과 support를 제대로 구분

### causal history

Zn-ion features의 slope는 주로 1~3개 직전 간격이다. noise에 민감할 가능성이 있으므로 train 내부에서 noise 대비 rate 변동을 계산한 뒤 긴 window를 추가한다. 처음부터 Transformer를 붙일 필요는 없다.

- 실제 시간 간격을 사용하는 짧음/중간/긴 window slope, dispersion, curvature.
- 초기 이력에는 valid length, padding mask, 실제 elapsed time을 전달한다.
- 같은 feature의 작은 MLP를 기준으로 small causal TCN 또는 GRU 하나를 비교한다.
- 미래를 바꾸었을 때 과거 feature/예측이 바뀌지 않는 causal test를 필수로 둔다.
- online RUL에서는 현재까지 관측을 사용한다. prefix-only forecast는 별도 task이며, 미래 실제 health를 입력할 수 없다.

### support distance와 regime novelty

`model.py`의 min/max distance는 box distance이다. convex hull distance와 같지 않다. 또한 `presets.py`의 feature index 2는 window heterogeneity 좌표이며 이것 자체가 hull distance라는 증거는 없다.

따라서 `d_support`(train 기하학적 이탈), `z_regime`(rate/curvature 변화), `u_error`(예측 오차 위험)를 따로 계산·보고한다. decay를 적용할 때 **거리가 멀면 affine이 더 옳다**는 가정을 별도로 확인해야 한다. affine prior가 틀린 구간에서 correction을 줄이면 더 악화될 수 있다.

지원 geometry는 고정 저차원 train 좌표의 hull, kNN distance, 기존 box distance를 비교한다. scaling/PCA는 train에만 fit한다. 이후 distance shell별 affine와 residual의 상대 오차를 측정한다. 높은 차원에서 hull-out 비율만 높다는 이유로 어려운 외삽이라고 판정하지 않는다.

## 6. P3 후보: dynamic gate는 최소 모델을 이긴 뒤 채택

앞 문서 `PP_MODELING_AUDIT_AND_NEXT_ARCHITECTURE_KO.md`의 두 expert 설계는 장기 후보로 해석한다. 현재 unit 수에서 처음부터 scale head+local head+gate+uncertainty를 공동 학습하는 것은 원인 분리를 어렵게 한다.

단계:

1. normalized BQ 한 경로.
2. 한 residual의 상대 scale correction.
3. 두 경로가 실제 서로 다른 unit/시점에서 유리함이 OOF로 확인될 때만 gate.

Gate 목표는 같은 unit을 보지 않은 두 expert의 OOF loss 차이에서 만든다. 최종 test의 승패를 gate label로 만들지 않는다. pseudo knee score는 관측된 변화의 보조 target이지 미래 knee 정답이 아니다.

Gate 입력은 현재까지의 history, valid length, train support 거리다. 미래를 보지 않는 smoothing 또는 변화 증거가 약할 때의 temporal penalty를 사용할 수 있다. entropy floor는 optional ablation이다. 하나의 경로가 늘 더 좋은 상황에서 억지로 두 경로를 균등 사용하게 하면 성능이 낮아진다.

같은 관측 prefix 뒤 미래가 다르게 진행되는 경우는 gate를 크게 해도 정확히 구분할 수 없다. 추가 operating context, 현재 관측, 또는 분포 예측이 필요하다. 이를 NN 용량 부족으로만 처리하지 않는다.

## 7. 불확실성: disagreement와 error risk를 구별

`oof_uncertainty.py`의 target은 teacher residual의 seed 표준편차다. 여러 seed가 같은 bias를 가지면 값이 작을 수 있다. 또한 head checkpoint는 head의 학습 loss로 선택한다. 후속 구현은 head 평가용 unit을 별도 분리해야 한다.

OOF absolute error 또는 squared error를 예측하는 risk head를 추가 대조군으로 둔다. 이것은 순수 epistemic uncertainty가 아니라 관측된 총 오차 위험의 proxy다. disagreement head와 구별해서 이름 붙인다. fold별 target_scale이 다르면 공통 물리 단위 또는 사전에 정한 공통 normalized 단위로 target을 맞춘다.

구간을 추가할 경우 unit별 calibration split과 distance shell별 empirical coverage/width를 보고한다. 외삽에서 일반적인 calibration 가정이 자동으로 유지되는 것은 아니므로 보장 문구는 증명한 조건 안에서만 사용한다.

## 8. 후속 실행 계약

### 기록과 split

모든 run은 다음을 보관한다: dataset/physical-unit ID, 원본 해시, 시간 및 EOL 정의, feature/label 단위, train/inner-val/outer-val ID, history cutoff, seed, config, selected epoch, git SHA. 값이 큰 결과를 보고 test split을 교체하지 않는다.

모델 선택용 inner unit, 개발 성능 평가용 outer unit을 분리한다. 장수명 unit을 제외하면 학습에 장수명 사례가 사라지는 fold도 그대로 보고한다. fold마다 scaling, affine, prototype memory, gate target을 다시 만든다. 평가 unit의 EOL을 memory에 넣지 않는다. 최종 refit은 PP와 baseline 모두 동일하게 수행한다.

### 최소 arm과 예산

| Arm | 변경 | 반드시 같은 조건으로 비교할 항목 |
|---|---|---|
| A0 | 현재 frozen/developed PP 재현 | 기존 artifact와 row ID·prediction 대응 |
| A1 | normalized affine BQ only | A2 residual의 순수 기여 |
| A2 | normalized additive BQ | A0 및 같은 입력 direct MLP |
| A3 | multiplicative BQ | A2와 encoder/optimizer 예산 동일 |
| A4 | A2 또는 A3 + causal history | 같은 history를 받은 MLP/선택 temporal baseline |
| A5 | A4 + OOF dynamic gate | gate 없이 가장 좋은 한 경로 |
| A6 | A5 + risk shrinkage | risk head 없는 A5 |

모든 arm을 처음부터 실행할 필요는 없다. 이전 단계의 unit validation 이득이 없으면 멈추고 실패 원인만 저장한다. screen은 seeds 42/43, 최종 비교는 42–46을 제안한다. PP와 baseline의 hyperparameter trial 수와 최대 epoch을 맞추고 실제 학습 시간도 기록한다. 데이터별로 잘 나온 seed만 표에 쓰지 않는다.

### 데이터 배치

- Zn-ion: residual-scale 병목과 장수명 의존성의 1차 개발 진단.
- Na-ion: 동일 boundary prior 아래 chemistry 이동의 회귀 확인.
- Sunwoda/RWTH/MICH: 기존 공동 dual-scale PP의 정확도/강건성 tradeoff 확인.
- HUST/MATRb2: transport를 뺀 모델과 포함한 모델을 나눠 구조 기여 확인.
- NASA/Virkler/N-CMAPSS: 각 기존 executor에서 실제로 적용 가능한 공통 변경만 비교. 알려진 health failure boundary가 없으면 BQ를 강제로 적용하지 않는다.
- XJTU/FEMTO/milling: 실패 데이터 삭제 없이 diagnostic rows로 유지. endpoint 라벨, context 부족, 관측 가능한 health 좌표를 먼저 확인한다.

### 선택과 중단 기준

주 선택은 개발 outer 결과가 아니라 **inner validation pooled R²**로 수행한다. Outer 결과는 선택 절차 전체를 평가한다. 그 결과로 다시 설계하면 해당 outer는 개발 이력으로 남는다.

다음 권장 기준은 실행 전에 JSON으로 확정한다: 주 metric pooled R², 보조 unit error와 seed dispersion, 허용 회귀 폭(예시 0.01 R²), 예산 및 종료 조건. 0.01은 보편적인 통계 기준이 아니라 사용자 요구에 맞춰 고정할 실용적 margin이다. 작은 점수 차이는 unit paired CI를 함께 보아 동률 가능성을 보고한다. unit이 매우 적으면 CI가 좁아도 강한 일반화 근거로 사용하지 않는다.

## 9. 수정 위치와 산출물

| 위치 | 후속 수정 |
|---|---|
| `experiments/znion_bq_confirmatory.py` 및 Na-ion adapter | target scale과 output inverse transform, EOL/time 단위 명시 |
| `src/pp_extrapolation/boundary_quotient.py` | additive/multiplicative 옵션, affine RUL-space fitting, component diagnostics |
| `src/pp_extrapolation/model.py` | box distance 명명, clipping 정책과 activation 보고 |
| `src/pp_extrapolation/presets.py` | feature index 대신 명시적인 schema 확인; 새 preset은 별도 version |
| `src/pp_extrapolation/oof_uncertainty.py` | error-risk target, head의 unit validation |
| 새 `experiments/pp_scale_contract_audit.py` | adapter별 단위·mB·실제 보정 폭 감사 |
| 새 `experiments/pp_scale_aware_nested_benchmark.py` | A0–A3 먼저, 단계별 재현 가능 runner |

산출물은 `manifest.json`, `selection.json`, `results.json`, unit/time/seed가 들어간 predictions, loss history, 각 arm 탈락 이유가 담긴 Markdown이다. 기존 최종 표는 신규 실험 완료 전에 바꾸지 않는다. 다른 저장소의 PAE adapter 의존성은 경로와 revision을 고정하거나 PP 쪽으로 필요한 데이터 adapter만 분리한다.

## 10. 논문 기여의 후보와 한계

가장 일관된 가설은 **사용 가능한 boundary/affine prior를 유지하면서, 단위가 정합적인 보정 폭을 관측 이력과 검증된 신뢰도에 맞게 조절하면 외삽 bias와 재학습 분산의 tradeoff가 개선된다**이다.

증명 가능한 구조 성질은 boundary zero, additive/relative correction bound다. 성능 우위는 실험으로 확인해야 한다. scale normalization, gated expert, positive output 각각의 사용만으로 새 논문 기여가 검증된 것은 아니다. 새로움의 문헌 검증은 별도 작업이며 이 문서는 신규 문헌 검색을 수행한 리뷰가 아니다.

**후속 모델에게 줄 첫 지시:** P0 및 A0–A4 결과를 재실행해 artifact를 확인한 뒤, full-development refit을 PP와 강한 baseline 모두에 동일 적용하라. 다음 신규 cohort에서는 refit 모델을 동결해 평가한다. scale-aware 후보는 장수명 unit 부재 fold가 해결되기 전 재채택하지 않는다. 이후 독립 장수명 development unit이 확보되면 A2/A3와 dynamic gate를 다시 검토한다.
