# PP-X 논문 최종 서사 · 약점 · Evidence Registry

작성자: 박진서  
작성일: 2026-09-12  
상태: 제출 전 고정 문서  
관련 커밋: `5c321ef` (Selective equal-coverage), `72104ea`/`505f7d3` (cross-domain stability)

---

## 0. 한 줄 정의

> **PP-X is a contract-conditioned, validation-approved prior-residual execution architecture, not a universal neural predictor.**

“새로운 neural architecture”로 강하게 밀지 않는다. 본문은 **assumptions의 선언·승인·실행·거절**을 하나의 검증 가능한 절차로 만든 점을 중심으로 쓴다.

---

## 1. 남아 있는 핵심 약점

### 1.1 구조 최초성은 중간

이번 업로드는 증거를 강화했지, PP-X의 기본 예측식을 완전히 새로운 구조로 바꾸지 않았다.

| 구성요소 | 기존 인접 개념 | PP-X에서의 역할 |
|---|---|---|
| prior | hybrid RUL / affine–physics baseline | frozen extrapolative path |
| bounded residual | residual correction / clipped NN | source-supported deviation만 학습 |
| support-aware scaling | applicability domain / distance gate | optional executor |
| validation-based route | model selection / selective prediction | unit-risk approval + exact fallback |

각 부품의 단독 최초성을 주장하지 않는다. 주장 가능한 것은 **typed contract → prior-residual core → unit-evidence approval → frozen execution**의 결합이다.

### 1.2 Evidence registry가 없으면 데이터 지위가 혼동됨

Cross-domain stability 결과는
[`PPX_DOMAIN_STABILITY_RESULTS_KO.md`](PPX_DOMAIN_STABILITY_RESULTS_KO.md)와
별칭 [`PPX_CROSS_DOMAIN_STABILITY_COMPARISON_KO.md`](PPX_CROSS_DOMAIN_STABILITY_COMPARISON_KO.md)
에 있다. 본 문서 §3의 registry table이 **데이터셋 지위·unit·coverage·route·fallback·selective 비교**의 단일 기준이다.

### 1.3 평균 성능 SOTA는 말하지 않음

Equal-coverage 비교와 8/9 equal-budget 우세가 있어도, 논문 중심은 “모든 데이터에서 최고 RMSE”가 아니다. 허용 결론:

1. 동일 coverage에서 PP-X가 더 나은/더 안전한 예측을 제공하는 조건이 있다.
2. 다른 설정에서는 prior route를 거절·fallback한다.
3. 이 조건부 실행이 unrestricted residual 또는 generic selective regression보다 안전하다.

금지 결론:

- 보편적 SOTA
- 모든 미래 cohort에서 실패 방지
- “새로운 NN architecture”로서의 단독 최초성

---

## 2. 논문에 가져갈 최종 서사

### 2.1 제목 후보

1. **When to Trust Structural Priors: Validation-Approved Prior-Residual Extrapolation for Prognostics**
2. **Validation-Approved Prior-Residual Execution for Safe and Selective RUL Extrapolation**

영문 방법 표현(초록·기여문):

> A validation-approved prior-residual framework for contract-conditioned extrapolation

### 2.2 기여 3개로 고정

기존 4개 기여 초안(`PPX_PAPER_CONTRIBUTIONS_KO.md`)은 참고용으로 유지하되, **제출 본문은 아래 3개로 압축**한다.

#### C1. Contract-conditioned model

관측 가능한 경계, 시간 이력, support, causal 정보로 허용 가능한 structural prior를 선언하고, frozen prior 주변에서 bounded causal residual을 학습하는 PP-X 구조.

#### C2. Validation-approved execution

residual bound, dual scale, transport, history 같은 executor를 전역적으로 켜지 않고, validation evidence가 해당 shift 기제를 지지할 때만 실행하며 반례에서는 거절/fallback.

#### C3. Equal-coverage and unit-level evidence

동일 coverage selective regression 비교, unit-based bootstrap/paired analysis, cross-domain stability, failure analysis, prospective evaluation을 통합해 “선택적 예측이라 성능이 좋아 보였다”는 설명을 검증.

이렇게 쓰면 **모형 구조 + 실행 정책 + 증거 패키지**가 각각 분리된다.

### 2.3 권장 Introduction 흐름

1. 외삽은 추가 가정 없이는 식별되지 않는다.
2. 기존 방법(CMNN, PINN, Engression, Xtrapolation, selective regression)은 각자 prior/가정 환경에서 강하다.
3. 남은 공백: 이질적인 prior를 outcome-free contract로 제한하고, unit risk로 executor와 fallback을 시험 전에 고정할 수 있는가.
4. 연구 루트: 식이 없거나 약하면 **PP-X**, 적용 가능한 식이 정당화되면 **PAE**(후속).
5. 오늘은 PP-X: C1–C3.
6. 구조 → Algorithm 1 → 프로토콜 → 결과 → 실패 분석 → 주장 경계.

---

## 3. Evidence Registry Table

### 3.1 지위 정의

| 지위 | 의미 | 본문에서의 사용 |
|---|---|---|
| **Main retrospective development** | 논문 메인 9-setting. route 개발에 열림 | 주 결과·ablation·equal-budget·stability |
| **Prospective / preregistered holdout** | 방법·임계값 동결 후 한 번만 공개 | route-selection evidence; 독립 predictive SOTA 아님 |
| **Limitation / retrospective extension** | 실패·복구·식별성 한계 | limitation 표. 메인 승수에 포함하지 않음 |
| **Mechanism / executor evidence** | CCMR v2.2 등 구조 검증 | 메인 모델 명칭이 아님 |
| **Rejected extension** | CRT, GCIE, v2.3 등 | 음성 결과로 공개 |

### 3.2 Main retrospective development (9 settings)

| Dataset | Units | PP-X R² | Equal-budget strongest | Route / executor | Fallback? | Alg.1 selective admissible coverage |
|---|---:|---:|---|---|---|---:|
| HUST | 16 | 0.958 | GroupDRO 0.955 | regime transport + core | No | 0.15% |
| Virkler | 10 | 0.888 | FT-Transformer 0.890 | support-gated residual core | No | 0% |
| NASA battery | 4 | 0.584 | Engression 0.583 | causal multiscale history | No | 25.5% |
| Sunwoda | 9 | 0.939 | linear-tail RBF 0.838 | fixed bounded BQ | No | 0% |
| RWTH | 8 | 0.878 | linear-tail RBF 0.732 | fixed bounded BQ | No | 0% |
| MICH | 8 | 0.751 | monotone NN −0.686 | dual-scale BQ | No | 0% |
| MATR2019 | 10 | 0.466 | FT-Transformer 0.342 | validation-calibrated core | No | 1.92% |
| MATR batch 2 | 9 | 0.862 | plain MLP 0.813 | transport + support decay | No | 1.23% |
| N-CMAPSS DS02 | 3 | 0.937 | Engression 0.932 | multiscale prior-residual | No | 100% |

합계: **77 physical units**. Equal-budget 8/9 win, exact sign \(p=0.0391\). Mixed strongest same-split은 별도 9/9 (\(p=0.00390625\))이며 섞지 않는다.

### 3.3 Prospective / holdout

| Dataset | 지위 | Route 결과 | Predictive 결과 | 문서 |
|---|---|---|---|---|
| N-CMAPSS DS03 | preregistered prospective | prior routes 거절 → direct fallback **PASS** | PP-X 0.882 < Engression 0.901 **FAIL superiority** | `NC_MAPSS_DS03_PPX_PROSPECTIVE_RESULTS_KO.md` |

### 3.4 Limitation / retrospective extension

| Dataset | 지위 | 비고 |
|---|---|---|
| XJTU | limitation; post-test development | 메인 승수 제외 |
| FEMTO | limitation; neural safety / abstention | ensemble만 약한 양수 |
| NASA milling | limitation; boundary-quotient post-test | 메인 승수 제외 |

### 3.5 Cross-domain stability (secondary)

| 모델 | macro R² | domain SD | domain MAD | R²>0 | vs PP-X exact MAD \(p\) |
|---|---:|---:|---:|---:|---:|
| **PP-X** | **0.807** | **0.174** | **0.061** | **9/9** | — |
| Engression | 0.257 | 0.949 | 0.280 | 7/9 | **0.0391** |
| GroupDRO | 0.011 | 0.998 | 0.619 | 5/9 | **0.0156** |
| plain MLP | 0.093 | 1.007 | 0.554 | 6/9 | **0.0469** |
| V-REx | 0.071 | 1.008 | 0.535 | 6/9 | **0.0469** |

8개 모델 동시 Holm 보정 후에는 비유의. 개별 대표 비교와 다중비교를 분리해 보고한다.  
상세: `PPX_DOMAIN_STABILITY_RESULTS_KO.md`, `PPX_CROSS_DOMAIN_STABILITY_COMPARISON_KO.md`

### 3.6 Equal-coverage Selective Regression

Noskov–Fishkov–Panov Algorithm 1 재현.

| 목표 coverage | 실제 평균 coverage | PP-X nRMSE | Selective NW | PP-X wins |
|---|---:|---:|---:|---:|
| 25% | 5.8% | **0.272** | 1.279 | **5/5** |
| 50% | 8.7% | **0.296** | 1.275 | **5/5** |
| 75% | 11.5% | **0.303** | 1.274 | **5/5** |
| 90% | 13.2% | **0.305** | 1.266 | **5/5** |

4/9 setting은 Algorithm 1이 전부 거절. 비교 가능한 5개에서 PP-X 우세(양측 exact \(p=0.0625\), \(n=5\)의 최소값).  
상세: `SELECTIVE_REGRESSION_EQUAL_COVERAGE_RESULTS_KO.md`

### 3.7 Mechanism / rejected

| 항목 | 지위 | 역할 |
|---|---|---|
| CCMR v2.2 | mechanism evidence | trajectory-domain executor 검증. 논문 메인 모델명 아님 |
| CRT / GCIE / CCMR v2.3 | rejected | 음성 결과 공개 |

---

## 4. 본문에서 쓸 결론 문장 (고정)

> Across nine retrospective strict-extrapolation settings, PP-X is not positioned as a universal accuracy SOTA. Instead, it maintains predictions under contract-conditioned prior-residual execution, rejects unsupported executors, and, at equal achieved coverage, yields lower accepted risk than a testing-based selective nonparametric regression baseline that abstains on most out-of-support rows. Prospective DS03 confirms correct prior rejection but not predictive superiority of the selected fallback.

한국어:

> PP-X는 모든 데이터에서 최고 정확도를 주장하지 않는다. 승인된 prior-residual 경로로 외삽 예측을 유지하고, 근거 없는 executor는 거절하며, 동일 coverage에서는 support 밖을 대부분 버리는 selective regression보다 낮은 accepted risk를 보였다. DS03에서는 prior 거절은 맞았지만 fallback의 정확도 우월은 확보하지 못했다.

---

## 5. 관련 파일 인덱스

| 내용 | 경로 |
|---|---|
| **본 서사·registry** | `PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md` |
| 최종 모델 정의 | `PPX_FINAL_PAPER_MODEL_KO.md` |
| 기여 초안(4항, 참고) | `PPX_PAPER_CONTRIBUTIONS_KO.md` |
| 상위저널 패키지 | `PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md` |
| Equal-budget 8/9 | `FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md` |
| Cross-domain stability | `PPX_DOMAIN_STABILITY_RESULTS_KO.md` |
| Stability 별칭 | `PPX_CROSS_DOMAIN_STABILITY_COMPARISON_KO.md` |
| Selective equal-coverage | `SELECTIVE_REGRESSION_EQUAL_COVERAGE_RESULTS_KO.md` |
| Selective 프로토콜 | `protocols/SELECTIVE_REGRESSION_EQUAL_COVERAGE_PROTOCOL.md` |
| DS03 prospective | `NC_MAPSS_DS03_PPX_PROSPECTIVE_RESULTS_KO.md` |
| PPT | `ppt/PP-X_Research_Detailed_v2.pptx` |

---

## 6. 실행 기록

2026-09-12에 본 문서를 생성하고, 위 registry와 3기여 서사를 제출용 기준으로 고정했다. PPT 흐름은 이미 사전조사 → 연구 루트 → 기여 → 구조 → 결과 순으로 재배치되어 있다. 추가 실험이 아니라 **문서·주장 경계 고정 작업**이다.
