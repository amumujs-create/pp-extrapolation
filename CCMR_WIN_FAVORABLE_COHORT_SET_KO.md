# CCMR이 비교군을 이길 가능성이 큰 고호트 셋

작성자: 박진서  
동결: 2026-09-11 (Stage-2 `TP_k*` 용량 outcome 미개봉)

## 한 줄 결론

**비교군(Engression 포함)을 이기려면 Alloy가 아니라 MultiStage형
레짐을 써야 한다.** 이미 열린 승기 고호트는 MultiStage RPT Stage-1 +
CCMR v2.0이고, 다음 미개봉 확증 후보는 같은 출처의 Stage-2 `TP_k*`다.

## 왜 이 셋인가

기존 봉인 결과에서 CCMR의 승패는 고호트 레짐에 크게 갈린다.

| 고호트 | CCMR 역할 | Engression 대비 | 셋 배정 |
|---|---|---|---|
| MultiStage RPT Stage-1 | 성능 성공; v2.0은 평균·max-regret 모두 우위 | v1.9 평균 동률·안전 우위; **v2.0이 평균도 우위** | **주 승기 셋** |
| Alloy A | 독립 성능 성공이지만 평균 정확도는 열세 | Engression R² 0.991 vs CCMR ~0.76–0.77 | **반례/정직성 부록** |
| Concrete / Luminosity / LG / SIT / RADAR | safe fallback 또는 실패 | 비교군 승자 주장 불가 | 개발·감사만 |

따라서 “다른 모델을 이긴다”는 문장은 Alloy+MultiStage를 묶어 쓰지 않고,
**MultiStage형(동일 셀화학·unit-disjoint·늦은 progress 외삽·수십 unit)**
에만 건다.

## 셋 구성

### A. Retrospective 승기 벤치마크 (이미 열림)

- 고호트: `MultiStage_RPT` (Stage-1 `TP_z*`, 적격 72 cell)
- CCMR: **v2.0 causal dynamics bank**  
  (`results/ccmr_v20_frozen_holdout_replay/predictions.npz`의
  `MultiStage_RPT_CCMR_v20`)
- 비교군: V-REx, GroupDRO, Monotone NN, Linear-tail RBF, Engression,
  linear-mean GP, TabPFN v3  
  (`experiments/ccmr_winset_multistage_v20_competitors.py`)
- 주장 가능 범위: frozen holdout replay / retrospective.  
  “신규 미개봉 확증”이라고 쓰지 않는다.

근거 수치(이미 기록됨):

- Persistence R² 0.973
- Engression R² 0.979, max regret 205%
- CCMR v2.0 R² 0.989, pooled/macro 개선 36%/50%, **max regret 0%**

### B. Prospective Stage-2 (1차·나머지 실행 완료)

1. `multistage_stage2_tpk_ccmr_v20`  
   - 인구: `TP_k*` 66 cell (Stage-1과 ID 비겹침)  
   - 결과: **모형 적합 전 inconclusive** (RPT≥10 적격 12개)  
   - 기록: `MULTISTAGE_STAGE2_TPK_CCMR_V20_RESULTS_KO.md`

2. `multistage_stage2_tpz_ccmr_v20` (나머지)  
   - 인구: `TP_z*` 72 cell, 적격 60  
   - **Stage-1과 동일 물리 cell** → 독립 고호트 아님  
   - 결과: validation `stable_bank` 통과 후 test **성능 성공 아님**  
     (R² -0.34, pooled 개선 0.08%, max regret 0%)  
   - 기록: `MULTISTAGE_STAGE2_TPZ_CCMR_V20_RESULTS_KO.md`

### C. 명시적 제외

다음을 이 승기 셋에 넣지 않는다.

- Alloy A를 “비교군 전체 승자” 문장에 포함
- condition/material shift 고호트(Concrete, Luminosity, Perovskite)
- 소표본·eligibility 탈락 후보(resistor2, metalwear, Oxford 8-cell)
- MATR/BatteryLife PP dual-scale 라인 (CCMR 계약과 다름)

## 성공 문장 템플릿

허용:

> MultiStage형 unit-disjoint 시간 외삽에서 CCMR v2.0은 Engression 대비
> 평균 RMSE를 개선하고 max unit regret 0%를 유지했다.

금지:

> CCMR이 모든 고호트·모든 비교군에서 최고다.
> Alloy에서도 Engression을 이겼다. (사실이 아님)

## 레지스트리

기계판독 정의: `protocols/CCMR_WIN_FAVORABLE_COHORT_SET.json`

Counted sealed limit(3)은 그대로 둔다. Stage-2 `TP_k`가 성공해도
자동 counted 승격하지 않고, 별도 registry 개정 후에만 센다.
