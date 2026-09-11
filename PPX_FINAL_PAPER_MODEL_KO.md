# 논문용 최종 PP-X 모델 정리

## 결정

PP-X를 “모든 optional module을 한 번에 켠 거대 모델”로 쓰지 않는다. 최종 논문 모델은 **작은 공통 core와 validation-approved executor**로 정의한다. 이 정리는 최종 ablation의 유의성 결과를 반영한다.

논문 Algorithm 1의 동결 구현은
`src/pp_extrapolation/paper_ppx.py::select_paper_ppx`이며, 입력·후보·임계값·
동률 규칙·fallback은 `protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md`에
고정한다. 이 함수는 선택정책을 하나의 실행 가능한 인터페이스로 만든 것이며,
12개 과거 route가 하나의 동일 신경망이었다는 뜻은 아니다.

## 공통 core: 모든 승인 prior 경로에 유지

각 시점의 causal adapter가 현재 상태, 짧은 history, 열화율, context, support feature를 만든다. 알려진 boundary 또는 source OOF에서 승인된 tail prior를 먼저 계산하고, frozen affine/quotient path 주위의 nonlinear residual을 학습한다.

\[
  \hat y = D\{y_{prior}+b(z)\tanh[r(z)/b(z)]\}.
\]

여기서 `D`는 RUL의 비음수 등 domain contract를 강제하는 decoder다. affine path는 train에서 결정한 뒤 frozen이며, test label·test batch 통계·test-time adaptation은 사용하지 않는다.

논문에서 core의 직접 근거로 쓰는 것은 nonlinear residual이다. 이 요소는 Sunwoda, RWTH, MICH에서 모두 unit-bootstrap과 BH 보정 후 개선을 보였다. Frozen affine은 모든 세 setting에서 점수 방향은 양수지만 독립 유의성은 확보하지 못했으므로, “입증된 단독 성능 요소”가 아니라 tail의 식별성과 경계 준수를 위한 parameterization으로 설명한다.

## 선택형 executor: 공용 기본값에서 제외

| Executor | 켜는 조건: train/validation evidence만 사용 | ablation 판정 | 논문에서의 위치 |
|---|---|---|---|
| fixed residual bound | group-disjoint validation에서 unbounded residual보다 낮은 loss | Sunwoda 강한 개선, RWTH 방향성, MICH 악화 | boundary cohort용 선택 모듈 |
| support-adaptive dual scale | fixed bound보다 validation 개선이 있고 support heterogeneity가 높음 | MICH 강한 개선; RWTH 유의한 악화 | MICH형 relationship-shift executor |
| regime transport | group-LOO validation에서 raw prediction보다 일관된 개선 | HUST·MATR-b2 강한 개선 | cohort shift executor |
| multiscale history | validation loss가 basic/short history보다 개선 | NASA·N-CMAPSS 소폭 개선, 통계력 부족 | adapter hyperparameter |
| neural safety | prior evidence가 부족하거나 source OOF prior regret가 양수 | FEMTO는 통계적으로 미확증 | abstention/fallback, 성능 주장의 핵심 아님 |

따라서 dual-scale, transport, full multiscale history를 “PP의 항상 켜진 구성요소”라고 쓰지 않는다. 특히 RWTH에서 dual-scale을 일괄 적용하면 유의하게 악화됐으므로, 전역 default로 둘 수 없다.

## 실행 규칙

1. **Domain contract 선언:** boundary의 존재, unit/group 정의, causal adapter, 허용 prior를 train 전에 선언한다.
2. **Common core fitting:** train만으로 prior와 nonlinear residual을 학습한다.
3. **Executor selection:** validation 또는 group-LOO validation에서 후보 executor를 고른다. 선택 기준과 동률 규칙은 test 전에 고정한다.
4. **Source evidence gate:** complete source group 수, regime coverage, OOF prior regret로 prior 자체를 승인하거나 neural safety로 보낸다.
5. **Frozen final evaluation:** 선택된 구조와 hyperparameter를 고정한 뒤 test를 한 번 예측한다.

이것은 mixture-of-experts나 test-time routing이 아니다. route와 executor는 source/validation evidence로 한 번 정해지고 test에는 frozen forward만 실행한다.

단순히 validation RMSE가 낮은 PP를 고르는 규칙은 12개 retrospective setting에서
7/12만 맞고 false accept 4개를 냈다. 따라서 Algorithm 1은 outcome-free typed
contract로 후보를 제한하고, 2% validation gain뿐 아니라 validation physical-unit
win fraction 60%와 worst-unit RMSE ratio 1.10을 함께 요구한다. 이 강화 정책도
contract admissibility를 고정 승인한 공통백본 감사에서는 false accept 2개가
남으므로, prospective validity가 확정됐다고 쓰지 않는다.

## 최종 주장 문장

> PP-X is a contract-conditioned prior-residual architecture: a frozen extrapolative prior provides the tail direction, a nonlinear residual learns source-supported deviations, and validation-approved executors handle only the shift mechanisms evidenced in source groups.

피해야 할 주장:

- “모든 prior와 모든 module이 모든 RUL 데이터에 유효하다.”
- “FEMTO fallback이 prior의 성능을 입증한다.”
- “5 random seeds가 독립 cohort 유의성을 제공한다.”

## 결과 보고 방식

본문에는 다음 세 표를 분리한다.

1. **Core ablation:** affine-only, core PP, direct NN. Nonlinear residual의 강한 근거를 제시한다.
2. **Executor ablation:** bound, dual scale, transport, history를 적용 가능한 setting에만 제시하고 승인 조건을 함께 쓴다.
3. **Competitor benchmark:** final selected PP-X와 강한 경쟁모델을 동일 split에서 비교한다. 유의성은 row가 아니라 physical unit bootstrap으로 보고한다.

FEMTO, XJTU, milling처럼 retrospective route 또는 낮은 식별성 setting은 applicability/limitation 표에 두며, main superiority 평균에 섞지 않는다.

## 기각된 generic residual transport 확장

Prior 거절 시 일반 외삽을 강화하려고 residual-only stochastic transport와
depth-shell gate를 추가로 평가했으나 논문 구조로 승격하지 않았다. DS03의
validation-only 연속 mass gate는 direct fallback보다 소폭 개선됐지만
Engression보다 낮았고, MultiStage에서는 안전성을 확보한 gate도 CCMR 단독보다
낮았다. 따라서 현재 Algorithm 1은 이 확장을 포함하지 않으며 prior evidence가
부족할 때 기존 fallback/abstention을 유지한다. 상세한 음성 결과와 재현 경로는
`PPX_CRT_REJECTED_EXPERIMENT_KO.md`에 기록한다.

CRT 이후 support-grade 조건부 implicit generator, group-CVaR energy loss,
unit-disjoint nested pseudo-extrapolation, source-only global gate를 결합한
GCIE도 평가했다. DS03에서 direct fallback 개선 신호는 있었지만 equal-budget
Engression과 2% worst-unit regret 기준을 동시에 넘지 못했다. 이 구조 역시
Algorithm 1에서 제외하며 상세 결과는
`PPX_GCIE_REJECTED_EXPERIMENT_KO.md`에 보존한다.

## 근거와 산출물

- 최종 on/off 및 BH 보정: `results/ppx_final_ablation_statistics_v1/REPORT_KO.md`
- 구성요소별 그림: `figures/paper/ppx_final_ablation/`
- 경쟁모델 그림: `figures/paper/ppx_final_ablation/fig_competitor_score_matrix.png`
- 원래 framework 정의: `PPX_FINAL_MODEL_KO.md`
