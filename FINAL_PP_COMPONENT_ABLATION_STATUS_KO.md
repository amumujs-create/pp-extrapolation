# PP-X 구성요소 ablation 현황과 필수 실험

> **Canonical paper pointer:** 논문 주체는 PP-X다. 동결 Algorithm 1은
> `PPX_FINAL_PAPER_MODEL_KO.md`, 최신 통계·주장 경계는
> `PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md`를 우선한다. 아래 BQ-PP,
> final PP와 결과 경로는 당시 arm을 식별하는 역사적 라벨이며 소급 개명하지
> 않는다.

## 결론

PP-X는 단일 고정 구조가 아니라 typed contract와 validation evidence에 따라
executor를 승인하는 구조다. 따라서 모든 데이터셋에 모든 모듈을 억지로 적용하는
full factorial보다, **공통 backbone ablation + prior별 executor ablation +
evidence gate ablation**으로 나누는 것이 논문 주장과 맞다.

대표 prior executor에 대한 matched ablation과 MATRb2 support-decay×transport 2×2 실험을 완료했다. 상세 결과는 `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md`에 있다. 새 untouched cohort 확증은 아직 필요하다.

## 이미 완료된 ablation

| 질문 | 비교 | 현재 근거 | 판정 |
|---|---|---|---|
| affine path만으로 충분한가 | affine quotient only vs BQ-PP | Sunwoda 0.340→0.939, RWTH 0.575→0.878, MICH −3.036→0.468 | NN residual 필요 |
| affine을 동결해야 하는가 | frozen vs trainable affine | Sunwoda 0.939 vs 0.900, RWTH 0.878 vs 0.855, MICH 0.468 vs 0.319 | 동결이 3/3 우세; 20/25 unit 승, p=0.071 |
| 어떤 history가 필요한가 | margin, margin-history, cycle, rate-history | 데이터셋별 feature ablation 완료 | full rate-history는 Sunwoda·RWTH, 단순 history는 MICH에 유리 |
| residual gain을 무조건 쓰면 되는가 | unconditional vs validation group-robust gate | Sunwoda와 Virkler에서 무조건 적용 시 악화 | evidence gate 필요 |
| causal temporal residual이 MATRb2를 개선하는가 | final PP vs temporal PP | 0.862 vs 0.860 ensemble | 현 GRU residual 기각 |
| support 밖에서 residual 수축이 필요한가 | decay off/on × transport off/on | MATRb2 5-seed 2×2 완료 | 단독 +0.002, SD 감소; transport와 결합 시 +0.042 |

## 완료된 최종 matched ablation 설계

공통 encoder, parameter budget, optimizer search budget, seeds `42–46`을 맞춰 다음 순서로 비교했다. prior별로 적용 가능한 arm만 실행했다.

| Arm | 제거하거나 추가하는 요소 | 검증하는 주장 |
|---|---|---|
| A | direct NN | 구조 prior가 없는 일반 NN 기준 |
| B | affine path only | 선형 tail만으로 설명되는 성능 |
| C | affine + unbounded NN residual | residual bound의 독립 효과 |
| D | frozen affine + bounded residual | PP 핵심 parameterization |
| E | D + support-distance decay | hull 밖 residual 수축 효과 |
| F | E + causal multiscale/history | 시간 이력의 추가 효과 |
| G | F + validation-only regime transport | cohort 출력 관계 보정의 추가 효과 |
| H | G + evidence approval gate | 불필요한 모듈을 거부해 성능을 보존하는 효과 |

## 데이터셋 배치

- **HUST**: protocol shift와 rate-conditioned transport를 검증한다.
- **NASA battery**: causal multiscale history 효과를 검증한다.
- **MATR batch 2**: regime transport와 seed 안정성을 검증한다.
- **N-CMAPSS**: 다변량 history와 운전조건 hull 외삽을 검증한다.
- **MICH**: boundary quotient가 health–RUL relationship shift를 복구하는지 검증한다.
- **Virkler**: 복잡한 history module이 필요 없는 경우 gate가 단순 PP를 유지하는지 검증한다.

Prior가 정의되지 않는 데이터에 boundary quotient를 켜는 비교는 모델 주장과 맞지 않으므로 `N/A`로 처리한다. 새 cohort에서 prior-availability rule 자체를 고정 검증하는 일은 확증 단계로 남는다.

## 보고 지표

각 arm에 대해 다음을 함께 보고한다.

1. pooled R²의 5-seed mean±SD와 prediction ensemble R²
2. unit-macro R²와 RMSE
3. convex-hull distance shell별 R²와 RMSE
4. full PP 대비 paired unit bootstrap CI와 seed 승패
5. 선택된 모듈, validation 개선률, 승인·거절 결정

## 기존 근거 파일

- `BOUNDARY_QUOTIENT_PP_RESULTS_KO.md`
- `BQ_PP_NOVELTY_AUDIT_KO.md`
- `PP_UNIVERSAL_IMPROVEMENT_AUDIT_KO.md`
- `MATR_BATCH2_TEMPORAL_PP_RESULTS_KO.md`
- `results/bq_pp_matched_controls_v1/results.json`
- `results/boundary_quotient_feature_ablation_v1/results.json`

동일 조건으로 다시 집계한 실제 수치와 해석은 `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md`에 정리했다.

논문 주장은 “모든 부품이 모든 도메인에서 항상 도움된다”가 아니다.
**outcome-free contract가 허용한 prior에 맞는 executor를 구성하고, validation
evidence가 부족하면 거부 또는 fallback한다**는 PP-X 주장을 ablation으로
검증한다.
