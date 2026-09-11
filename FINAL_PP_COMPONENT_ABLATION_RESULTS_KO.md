# PP-X 구성요소 ablation 결과

> **Canonical paper pointer:** 이 문서는 PP-X의 retrospective
> core/executor mechanism evidence다. 동결 Algorithm 1과 최신 유의성 판정은
> `PPX_FINAL_PAPER_MODEL_KO.md`,
> `results/ppx_final_ablation_statistics_v1/REPORT_KO.md`,
> `PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md`를 우선한다. BQ-PP와 final PP는
> 기존 실험 arm의 역사적 라벨로 보존한다.

## 실험 조건

Boundary-Quotient PP가 적용되는 Sunwoda·RWTH·MICH를 대상으로 동일 입력 11개, hidden width 64, affine alpha 1000, learning rate 0.001, weight decay 0.01과 seeds 42–46을 사용했다. soft-boundary weight만 validation에서 선택했다. `평균±SD`는 개별 seed pooled R²이고 `ensemble`은 다섯 예측을 평균한 pooled R²다.

## 논문 본문용 최종 기능별 ablation

아래 표는 PP-X의 contract-admissible executor 기능이 무엇을 개선했는지 보여주는
retrospective ablation 표다. `ΔR²`는 해당 기능을 켠 모델의
prediction-ensemble pooled R²에서 matched 제거 arm의 값을 뺀 것이다.

| PP-X executor 기능 | 대표 데이터셋 | 제거 arm → 기능 포함 arm | ΔR² | 확인된 역할 |
|---|---|---|---:|---|
| Nonlinear residual | Sunwoda | affine quotient 0.281 → bounded BQ 0.939 | **+0.658** | affine tail만으로 설명되지 않는 곡률 학습 |
| Nonlinear residual | RWTH | affine quotient 0.659 → bounded BQ 0.878 | **+0.219** | cohort별 nonlinear deviation 학습 |
| Nonlinear residual | MICH | affine quotient −3.343 → bounded BQ 0.468 | **+3.811** | relationship-shift에서 affine-only 붕괴 방지 |
| Frozen affine path | Sunwoda | trainable hard-boundary 0.900 → frozen BQ 0.939 | +0.039 | NN이 외삽 tail 방향을 덮어쓰는 현상 억제 |
| Frozen affine path | RWTH | 0.855 → 0.878 | +0.023 | tail 안정화 |
| Frozen affine path | MICH | 0.319 → 0.468 | +0.149 | 작은 cohort에서 baseline drift 억제 |
| Fixed residual bound | Sunwoda | unbounded 0.718 → bounded 0.939 | **+0.221** | 과도한 NN correction 억제 |
| Fixed residual bound | RWTH | 0.788 → 0.878 | +0.090 | 외삽 안정화 |
| Fixed residual bound | MICH | 0.759 → 0.468 | **−0.291** | 고정 bound가 필요한 correction까지 차단하는 반례 |
| Support-adaptive dual scale | MICH | fixed bound 0.468 → dual scale 0.751 | **+0.283** | 이질적 tail에서만 residual 용량 확대 |
| Causal rate history | Sunwoda | current margin 0.590 → full history 0.939 | **+0.349** | 현재 health가 같은 서로 다른 열화속도 구분 |
| Causal rate history | RWTH | −0.378 → 0.878 | **+1.256** | 속도·변동 이력이 핵심 상태 표현 |
| Causal rate history | MICH | margin history 0.715 → full history 0.468 | −0.247 | validation route 실패 반례, 단순 history가 유리 |
| Support-distance decay | MATRb2 | decay-off raw 0.674 → decay-on raw 0.675 | +0.002 | 평균 정확도보다 seed SD 0.217→0.148 감소 |
| Regime transport | HUST | raw 0.829 → transported 0.958 | **+0.128** | validation unit의 출력오차 구조 전달 |
| Regime transport | MATRb2 | decay-on raw 0.675 → transported 0.862 | **+0.187** | cell 간 state/rate scale 보정 |
| Decay×transport interaction | MATRb2 | transport-only 0.820 → decay+transport 0.862 | +0.042 | 먼 support에서 transport 과보정 완화 |
| Validation-selected causal route | NASA battery | short-only 0.572 → selected 0.584 | +0.012 | fold별 유효 history scale 선택 |
| Validation-selected multiscale route | N-CMAPSS | basic 0.928 → selected 0.937 | +0.009 | 운전조건과 열화의 시간척도 선택 |
| Evidence approval gate | 13 settings | always-on gain → gated gain | 3 개선 / 10 유지 / 0 악화 | 맞지 않는 optional executor 거절 |

### 이 표에서 분리해서 읽어야 하는 것

- 가장 일관된 핵심은 **frozen affine quotient + nonlinear residual**이다.
- Fixed bound와 full rate history는 조건부 기능이다. MICH 반례 때문에 universal default라고 주장하면 안 된다.
- Support-adaptive dual scale은 fixed bound의 MICH 실패를 복구하지만 Sunwoda·RWTH의 dataset별 최고점은 각각 0.005, 0.036 낮춘다. 대신 하나의 공동 모델에서 worst-domain R²를 0.468→0.751로 높인다.
- MATRb2에서는 support decay의 단독 정확도 효과가 거의 없고, transport와의 상호작용과 seed 안정화가 주효과다.
- Evidence gate는 새로운 예측 식이 아니라 optional PP 기능의 채택 규칙이다.

## Matched 6-arm 결과

| Arm | Sunwoda mean±SD / ens. | RWTH mean±SD / ens. | MICH mean±SD / ens. | 역할 |
|---|---:|---:|---:|---|
| Direct NN | −1.682±1.804 / −1.352 | 0.319±0.554 / 0.633 | **0.339±0.359 / 0.684** | 구조 prior 제거 |
| Soft-boundary NN | −3.635±3.854 / −3.107 | 0.312±0.586 / 0.483 | −0.334±0.664 / −0.057 | 경계 위반 penalty만 적용 |
| Fully trainable hard-boundary NN | 0.898±0.006 / 0.900 | 0.851±0.006 / 0.855 | 0.318±0.022 / 0.319 | exact boundary, affine 동결 제거 |
| Affine quotient only | 0.281±0.000 / 0.281 | 0.659±0.000 / 0.659 | −3.343±0.000 / −3.343 | NN residual 제거 |
| Frozen affine + unbounded residual | 0.689±0.141 / 0.718 | 0.760±0.047 / 0.788 | **0.532±0.217 / 0.759** | residual bound 제거 |
| **Frozen affine + bounded residual BQ-PP** | **0.909±0.020 / 0.939** | **0.871±0.016 / 0.878** | 0.468±0.000 / 0.468 | 현재 BQ executor |

## 구성요소별 결론

1. **Exact boundary만으로는 부족하다.** Soft-boundary NN은 세 데이터 모두 열세이고 seed 분산도 크다.
2. **Affine-only도 부족하다.** NN residual을 제거하면 Sunwoda 0.281, RWTH 0.659, MICH −3.343이다.
3. **Affine 동결은 유효하다.** bounded 조건에서 frozen BQ-PP가 fully trainable hard-boundary NN보다 세 데이터 모두 높다.
4. **Residual bound는 조건부로 유효하다.** Sunwoda와 RWTH에서는 unbounded 대비 각각 +0.221, +0.090이지만 MICH에서는 −0.291이다.
5. 따라서 bounded residual은 공용 상수가 아니라 validation evidence로 승인하는 executor 옵션이어야 한다.

25개 test unit의 paired RMSE에서 BQ-PP는 direct NN보다 17개 unit, soft-boundary NN보다 24개, affine-only보다 25개에서 이겼다. 양측 Wilcoxon p-value는 각각 `0.0028`, `1.13×10⁻⁶`, `5.96×10⁻⁸`이다. Fully trainable hard-boundary NN과의 비교는 20/25 unit에서 이겼지만 `p=0.071`이고, unbounded residual과는 14/25 및 `p=0.578`이므로 두 요소의 보편적 우월성은 확증되지 않았다. 이론적 bound `|prediction−affine|≤margin×B`는 17,645개 예측에서 위반 0건이었다.

## History feature ablation

| 입력 history | Sunwoda ensemble | RWTH ensemble | MICH ensemble |
|---|---:|---:|---:|
| Current margin only | 0.590 | −0.378 | 0.672 |
| Margin history | 0.550 | −0.197 | **0.715** |
| Margin history + cycle | 0.777 | −0.251 | 0.702 |
| **Full margin + rate history** | **0.939** | **0.878** | 0.468 |

Full rate history는 Sunwoda·RWTH에 필수지만 MICH에서는 단순 margin history보다 0.247 낮다. 기존 validation은 MICH에서도 full history를 선택했으므로, 현재 gate가 test의 새로운 health–RUL 관계 변화를 식별하지 못하는 반례다.

## 논문에서 사용할 ablation 주장

방어 가능한 주장은 다음과 같다.

> PP-X의 성능은 affine tail이나 hard boundary 하나에서 나오지 않는다.
> Frozen affine quotient와 nonlinear residual의 결합이 필요하며, residual
> bound와 causal rate history의 효용은 extrapolation regime에 따라 달라진다.
> 따라서 PP-X는 contract가 허용한 executor를 validation evidence로 승인한다.

“모든 PP 구성요소가 모든 데이터셋에서 항상 개선한다”는 주장은 결과와 맞지 않는다. MICH는 고정 bound와 full-history route의 반례이며, 이후 support-adaptive dual-scale PP가 이 실패를 0.751까지 복구했다.

## Support decay × regime transport: MATR batch 2

동일한 최종 PP 구조와 seeds 42–46에서 `residual_decay ∈ {0, 0.05}`와 validation-only transport on/off를 2×2로 비교했다.

| Support decay | Transport | 개별 pooled R² 평균±SD | prediction ensemble R² |
|---|---|---:|---:|
| Off | Off | 0.527±0.217 | 0.674 |
| On | Off | 0.556±0.148 | 0.675 |
| Off | On | 0.815±0.041 | 0.820 |
| **On** | **On** | **0.852±0.061** | **0.862** |

Support decay의 단독 ensemble 개선은 `+0.002`로 작지만 seed SD를 약 32% 줄였다. Transport의 효과가 가장 크며, decay와 함께 사용할 때 transport-only보다 ensemble R²가 `+0.042` 높다. 따라서 MATRb2에서는 decay를 독립적인 정확도 모듈보다 **transport가 멀리 있는 표본에서 과도하게 보정하지 않도록 안정화하는 상호작용 요소**로 해석한다.

## 대표 데이터셋별 prior executor ablation

| 데이터셋 | 제거한 요소 | 제거 시 ensemble R² | 최종/선택 모델 R² | 변화 |
|---|---|---:|---:|---:|
| HUST | rate-conditioned transport | 0.829 | **0.958** | +0.128 |
| NASA battery | validation-selected causal history | 0.572 short-only | **0.584** | +0.012 |
| MATR batch 2 | support decay + regime transport | 0.674 | **0.862** | +0.189 |
| N-CMAPSS | validation-selected multiscale feature route | 0.928 basic | **0.937** | +0.009 |
| Sunwoda | bounded frozen BQ residual | 0.718 unbounded | **0.939** | +0.221 |
| RWTH | bounded frozen BQ residual | 0.788 unbounded | **0.878** | +0.090 |
| MICH | bounded frozen BQ residual | **0.759 unbounded** | 0.468 bounded | −0.291 |

HUST와 MATRb2에서는 output transport가 큰 기여를 한다. NASA와 N-CMAPSS의 history 선택 효과는 양수지만 작다. BQ residual bound는 Sunwoda·RWTH에는 강하게 유효하지만 MICH에는 맞지 않는다.

이 반례를 해결하기 위해 local bound 2와 broad bound 6을 support heterogeneity로 전환하는 dual-scale PP를 적용했다. 동일 공동 모델의 ensemble은 Sunwoda 0.934, RWTH 0.842, MICH 0.751이며, 이전 통합 PP 대비 세 데이터 모두 개선됐다. 상세는 `UNIFIED_DUAL_SCALE_PP_IMPROVEMENT_KO.md`에 있다.

## Evidence gate ablation

13개 외삽 설정에서 validation MSE 2% 개선과 validation unit 80% 승리를 동시에 요구하는 gate를 적용한 기존 감사 결과는 `3개 개선, 10개 유지, 0개 악화`였다. gate 없이 residual gain을 항상 적용하면 Sunwoda `0.865→0.862`, N-CMAPSS `0.934300→0.934277`, Virkler deep-future `−5.374→−6.558`로 악화했다.

따라서 evidence gate의 역할은 평균 성능을 직접 만드는 것보다, 맞지 않는 executor를 거절해 base PP를 보존하는 것이다. 다만 MICH의 history route에서는 validation이 full history를 승인했는데 test에서는 simple history가 더 높았으므로 gate가 모든 relationship shift를 탐지한다고 주장할 수 없다.

## 남은 범위

공통 arm을 모든 데이터셋에 강제로 적용하는 실험은 prior availability가 다른 modular PP의 정의와 맞지 않는다. 현재 결과는 각 executor가 적용되는 대표 데이터에서 on/off 효과를 분리했다. 새 untouched cohort에서 prior-availability rule과 evidence gate를 고정 적용하는 확증은 여전히 필요하다.

재현 코드: `experiments/bq_pp_matched_controls.py`, `experiments/boundary_quotient_feature_ablation.py`, `experiments/matr_batch2_support_transport_ablation.py`  
원시 결과: `results/bq_pp_matched_controls_v1/`, `results/boundary_quotient_feature_ablation_v1/`, `results/matr_batch2_support_transport_ablation_v1/`
