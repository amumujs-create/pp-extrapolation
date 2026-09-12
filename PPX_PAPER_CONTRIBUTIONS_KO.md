# PP-X 논문 기여 문안

작성자: 박진서  
논문 메인: **PP-X — a validation-approved prior-residual framework for
contract-conditioned extrapolation**

> **제출용 고정 서사:** 기여는 3개로 압축한다.
> (`PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md`)
>
> 1. Contract-conditioned model  
> 2. Validation-approved execution  
> 3. Equal-coverage and unit-level evidence  
>
> 아래 4항 초안은 상세 초안이며, 본문 Introduction은 위 3항을 우선한다.
> PP-X는 universal neural predictor가 아니라 contract-conditioned,
> validation-approved prior-residual **execution architecture**로 쓴다.

## Introduction용 영문 Contributions

The contributions of this work are fourfold.

1. **Contract-conditioned formulation of extrapolation.** We formulate
   extrapolation as the conditional execution of structural assumptions,
   rather than the application of a single universal inductive bias. A typed
   contract declares which boundary, monotonicity, causal-history, support,
   and regime assumptions are admissible before test outcomes are observed.

2. **Prior-preserving residual architecture.** We introduce PP-X, which
   combines a frozen low-complexity extrapolative path with a nonlinear
   residual constrained to represent source-supported deviations. Optional
   residual bounds, support adaptation, multiscale history, and regime
   transport are treated as executors, not universally active components.
   Unsupported routes revert to a prespecified fallback or abstention.

3. **Physical-unit evidence for executor approval.** We provide a frozen
   selection procedure that evaluates candidate executors using
   group-disjoint validation evidence, physical-unit improvement, and
   worst-unit risk. This separates structural admissibility from empirical
   approval and prevents test-time routing or retrospective activation of a
   favorable module.

4. **Multi-domain evaluation with explicit negative evidence.** Across nine
   concept-aligned retrospective extrapolation settings, PP-X outperformed
   the strongest uniformly tuned 30-candidate baseline in eight settings
   (two-sided sign test, \(p=0.0391\)). A separate row-aligned analysis across
   77 physical units estimated a 33.8% equal-dataset geometric-mean RMSE
   reduction (hierarchical 95% CI: 15.8%–48.6%) against the available paired
   comparators. A preregistered N-CMAPSS DS03 replay correctly rejected
   unsupported prior routes, although Engression remained more accurate than
   the selected PP-X fallback. We retain this result, together with rejected
   CRT and GCIE extensions, to define the empirical boundary of the method.

## 압축형 영문 Contributions

> We contribute PP-X, a contract-conditioned prior-residual framework that
> (i) declares admissible extrapolation assumptions before test evaluation,
> (ii) preserves a frozen extrapolative path while learning only
> source-supported nonlinear residuals, (iii) activates optional executors
> using group-disjoint physical-unit evidence with exact fallback, and
> (iv) evaluates both successful and failed routes under multi-domain,
> equal-budget, and prospective protocols.

## 한국어 대응본

본 연구의 기여는 네 가지다.

1. **외삽의 contract-conditioned 정식화:** 외삽을 하나의 강한 prior를 모든
   데이터에 적용하는 문제가 아니라, test 결과를 보기 전에 선언한 경계,
   단조성, 인과 이력, support 및 regime 가정 중 근거가 있는 것만 실행하는
   문제로 정식화했다.

2. **Prior-preserving residual 구조:** 동결된 저복잡도 외삽 경로가 tail 방향을
   유지하고, 신경망은 source에서 지지되는 비선형 편차만 residual로 학습하는
   PP-X를 제안했다. Residual bound, support adaptation, multiscale history,
   regime transport는 항상 켜지는 모듈이 아니라 조건부 executor로 분리했다.

3. **Physical-unit evidence 기반 승인:** group-disjoint validation, 물리 unit
   개선률, worst-unit risk를 이용해 executor를 승인하는 동결 절차를 제시했다.
   구조적으로 허용되는 prior와 실제 데이터에서 승인된 executor를 분리하며,
   test-time routing이나 test 결과를 본 사후 모듈 선택을 허용하지 않는다.

4. **성공과 실패를 함께 포함한 다중 도메인 검증:** 동일하게 30개 후보를 탐색한
   8개 비교모델에 대해 PP-X는 9개 설정 중 8개에서 우세했다
   (\(p=0.0391\)). 별도의 row-aligned 77-unit 분석에서는 사용 가능한 paired
   비교군 대비 데이터셋 동일 가중 기하평균 RMSE가 33.8% 감소했다
   (계층 bootstrap 95% CI: 15.8%–48.6%). N-CMAPSS DS03 prospective replay는
   근거 없는 prior를 올바르게 거절했지만 Engression보다 높은 예측 정확도는
   확보하지 못했다. CRT와 GCIE의 기각 결과도 함께 공개해 방법의 적용 경계를
   명시했다.

## Abstract용 기여 문장

### 영문

> Rather than imposing a universal prior, PP-X declares an outcome-free
> structural contract, learns source-supported residual deviations around a
> frozen extrapolative path, and activates only those executors approved by
> group-disjoint physical-unit evidence. This design makes extrapolation
> assumptions explicit, test-independent, and rejectable.

### 한국어

> PP-X는 보편적인 prior를 강제하는 대신 outcome-free 구조 계약을 선언하고,
> 동결된 외삽 경로 주변의 source-supported residual을 학습하며, group-disjoint
> physical-unit evidence가 승인한 executor만 실행한다. 이를 통해 외삽 가정을
> 명시적이고 test-independent하며 거절 가능한 대상으로 만든다.

## Discussion용 방법론적 기여

PP-X의 핵심 기여는 새로운 개별 신경망 블록 하나가 아니다. Affine path,
bounded residual, validation selection, support distance는 각각 기존 개념과
겹칠 수 있다. 본 연구의 차별점은 다음 세 층을 하나의 검증 가능한 실행
절차로 결합한 데 있다.

- **Declare:** outcome-free typed contract로 허용 가능한 prior를 제한한다.
- **Learn:** frozen prior 주위에서 source-supported residual만 학습한다.
- **Approve or decline:** physical-unit validation evidence로 executor를
  승인하고, 근거가 부족하면 exact fallback 또는 abstention을 수행한다.

이 구조는 외삽에서 가정을 숨겨진 hyperparameter로 취급하지 않고, 사전에
선언하고 검증하며 거절할 수 있는 연구 대상으로 만든다.

## 주장하지 않는 기여

다음 문장은 논문 기여로 사용하지 않는다.

- PP-X가 모든 RUL 또는 OOD 데이터에서 보편적 SOTA라는 주장
- 모든 executor가 모든 데이터셋에서 성능을 높인다는 주장
- DS03가 PP-X의 prospective predictive superiority를 입증했다는 주장
- CCMR, CRT 또는 GCIE가 PP-X와 동의어라는 주장
- 5개 random seed를 독립 physical-unit 표본으로 해석하는 주장

## 기여와 근거 연결

| 기여 | 직접 근거 |
|---|---|
| Typed contract | `protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md` |
| Prior-residual core | `results/ppx_final_ablation_statistics_v1/REPORT_KO.md` |
| Executor conditionality | RWTH dual-scale 악화, MICH fixed-bound 악화 |
| Unit-evidence approval | `PPX_FINAL_PAPER_MODEL_KO.md` |
| Equal-budget 경쟁력 | `FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md` |
| Unit-level 효과 | `JOURNAL_EVIDENCE_COMPLETE_KO.md` |
| Prospective route 선택 | `NC_MAPSS_DS03_PPX_PROSPECTIVE_RESULTS_KO.md` |
| 적용 경계 | `PPX_CRT_REJECTED_EXPERIMENT_KO.md`, `PPX_GCIE_REJECTED_EXPERIMENT_KO.md` |
