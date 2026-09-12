# PP-X 상위저널 검증 패키지

작성자: 박진서  
논문 메인: **PP-X**  
방법 표현: **A validation-approved prior-residual framework for
contract-conditioned extrapolation**

> **제출 전 고정 서사·Evidence Registry:**
> [`PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md`](PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md)
> Cross-domain stability 별칭:
> [`PPX_CROSS_DOMAIN_STABILITY_COMPARISON_KO.md`](PPX_CROSS_DOMAIN_STABILITY_COMPARISON_KO.md)

## 1. 이번에 고정한 것

- 논문 알고리즘과 임계값:
  `protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md`
- 실행 가능한 선택 함수:
  `src/pp_extrapolation/paper_ppx.py`
- 동결 커밋: `03b3affc40a6e175538e17537e4778271a157f78`
- 논문 메인 evidence: 9개 retrospective development setting
- limitation tier: XJTU, FEMTO, NASA milling
- CCMR v2.2: 메인 정확도 모델이 아니라 risk-aware routing mechanism evidence

## 2. 교정된 최종 성능 증거

기존 paired 감사가 Sunwoda와 RWTH에도 MICH용 dual-scale executor를 적용하는
오류를 수정했다. 논문 최종 route는 다음과 같다.

- Sunwoda: fixed bounded BQ-PP, R² **0.939**
- RWTH: fixed bounded BQ-PP, R² **0.878**
- MICH: support-adaptive dual-scale BQ-PP, R² **0.751**

교정 후 9개 설정·77개 물리 unit:

- pooled R² wins: **9/9**
- dataset exact sign test: **p=0.00390625**
- equal-dataset mean log-RMSE ratio: **0.412**
- hierarchical 95% CI: **[0.172, 0.665]**
- geometric-mean RMSE reduction: **33.8%**
- 환산 95% CI: **15.8%–48.6%**
- dataset-level BH q<0.05: **Sunwoda, MATR-b2**
- RWTH unit bootstrap CI는 0을 제외하지만 BH q=0.0703

이 수치는 development-final retrospective 결과다. 9/9를 prospective
generalization probability로 해석하지 않는다.

## 3. Core·executor ablation

20개 row-aligned component comparison을 재실행했다.

강한 개선:

- nonlinear residual: Sunwoda, RWTH, MICH
- fixed residual bound: Sunwoda
- support-adaptive dual scale: MICH
- regime transport: HUST, MATR-b2
- complete selected PP-X vs direct NN: Sunwoda, RWTH

필수 반례:

- RWTH에서 dual scale은 악화
- MICH에서 fixed bound는 악화

따라서 optional executor를 모든 데이터에 켜는 architecture로 설명하지 않는다.
이 반례가 contract-conditioned selection의 직접 근거다.

## 4. 선택정책 공격에 대한 결과

공통 PP backbone과 matched direct model의 12-domain retrospective audit:

| 정책 | 정확 | false accept | false reject | 해석 |
|---|---:|---:|---:|---|
| Always direct | 6/12 | 0 | 6 | 안전하지만 이득 포기 |
| Always PP | 6/12 | 6 | 0 | 전역 prior 실패 |
| Validation RMSE only | 7/12 | 4 | 1 | validation-best 불충분 |
| Frozen PP-X unit-risk approval | 8/12 | 2 | 2 | 개선되나 contract 없이 완전 해결 아님 |
| Test oracle | 12/12 | 0 | 0 | 비배포 상한선 |

PP-X 정책의 equal-dataset log-RMSE CI도 0을 포함한다. 따라서 현재 결과로
“validation gate가 미래 domain route를 완전히 식별한다”고 주장하지 않는다.

남은 두 false accept:

- FEMTO: typed source contract에서 complete group 부족으로 prior를 거절해야 함
- NASA milling: common affine PP가 아니라 알려진 failure boundary에 맞는
  boundary-quotient 후보만 contract-admissible해야 함

즉 executor selection 전에 typed prior contract가 필요한 이유를 보여 준다.
다만 이 설명은 retrospective mechanism 해석이며 새 cohort 확증이 필요하다.

## 5. 통계 검증

- row가 아니라 physical unit을 추론 단위로 사용
- prediction ensemble을 주 결과로 사용
- unit sign-flip + unit bootstrap
- dataset–unit hierarchical bootstrap
- dataset family에 Benjamini–Hochberg 보정
- 5 seeds는 optimizer stability이며 독립 표본으로 취급하지 않음
- 상대 RMSE의 heavy negative tail 때문에 본문 주 효과는 대칭적인
  log-RMSE ratio로 고정

## 6. 누수·재현성 감사

- 선택 함수 API에 test outcome 인자 없음
- prior gate·executor gate·paper selector의 test 인자 부재 검사 통과
- contract-inadmissible 후보 전달 시 오류
- source evidence 부족 시 exact fallback
- 모든 12 registry artifact 존재 및 SHA-256 기록
- protocol introduction commit 기록
- 전체 package test는 별도 실행

감사 결과:
`results/ppx_paper_reproducibility_audit_v1/results.json`

## 7. 경쟁모형 공정성

구분해서 보고해야 한다.

1. **Strongest pooled comparator:** 같은 split에서 보고된 가장 높은 pooled R²
2. **Row-aligned paired comparator:** 저장된 동일 row·unit 예측으로 paired
   inference가 가능한 비교군

RWTH, MATR-b2, N-CMAPSS에서는 두 비교군이 다르다. pooled 우위와 paired
unit 통계를 한 모델의 결과처럼 합치지 않는다.

MICH raw-cycle에서는 29–32 validation candidates와 seeds 42–46의
V-REx, GroupDRO, monotone NN, Engression, full-train SVGP 비교가 완료됐다.
Milling은 validation group이 하나뿐이므로 stable ranking 근거에서 제외한다.

전체 9개 setting에서 plain MLP, FT-Transformer, V-REx, GroupDRO, monotone NN,
Engression, linear-tail RBF, full-train SVGP를 각각 정확히 30 validation candidate와
5 refit seed로 재학습했다. PP-X는 typed contract에 따라 후보 executor 종류가
달라지는 frozen framework이므로 하나의 공통 30-hyperparameter grid로 재개발하지
않았다. PP-X는 최강 동일예산 비교군에 8/9 pooled R² 우세였고 dataset sign test
양측 p=0.0391이었다. 상세는 `FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md`에 있다.

## 8. Prospective 감사

N-CMAPSS DS03의 test engine 6개를 미개봉 prospective cohort로 실행했다.
프로토콜, 원본 hash, selection artifact를 각각 test outcome 공개 전에 GitHub에
커밋했다. train-only prior evidence와 validation 결과가 prior-residual 후보를
지지하지 않아 frozen PP-X는 direct fallback을 선택했다.

- selected fallback prospective R²: 0.882
- basic PP R²: 0.832
- multiscale PP R²: 0.869
- strongest 30-candidate comparator: Engression 0.901
- route-selection success: PASS
- predictive-superiority success: FAIL

즉 미개봉 cohort에서 gate가 실제 최선 PP-X route를 골랐다는 미래 선택정책
증거는 확보했지만, 최강 모델보다 정확하다는 prospective 주장은 확보하지 못했다.
DS03는 DS02와 같은 N-CMAPSS 계열이므로 완전히 독립적인 실제 도메인 확증으로
과장하지 않는다.

## 9. 리뷰어 공격과 답변

| 공격 | 현재 답변 | 상태 |
|---|---|---|
| 데이터셋별 oracle 아닌가 | common backbone 6/12 실패와 반례를 공개; Algorithm 1 동결 | 부분 |
| validation-best일 뿐 아닌가 | RMSE-only 오탐 4 공개; typed contract + unit-risk 단계 분리 | 부분 |
| component가 항상 유효한가 | RWTH dual-scale, MICH fixed-bound 악화 공개 | 통과 |
| 행을 독립 표본으로 부풀렸나 | 77 physical-unit + hierarchical bootstrap | 통과 |
| 다중검정은 했나 | BH 보정; 2/9만 competitor paired q<0.05 공개 | 통과 |
| 표와 paired comparator가 다른가 | strongest pooled와 row-aligned paired 열 분리 | 통과 |
| 최종 route 통계가 맞나 | Sunwoda/RWTH routing 버그 수정 후 재집계 | 통과 |
| 모든 모델 예산이 같은가 | 9 setting × 8 baseline × 30 candidate × 5 refit 완료 | baseline 비교 통과 |
| prospective confirmation이 있나 | DS03에서 route 선택 성공, Engression 우월 | 부분 |
| 하나의 실행 알고리즘인가 | frozen selector API·protocol·tests 구현 | selection 층 통과 |

## 10. 논문에서 사용할 결론

> PP-X does not claim universal accuracy. It formalizes extrapolation as the
> conditional execution of structural assumptions: an outcome-free contract
> restricts admissible priors, group-disjoint validation approves one
> prior-residual executor, and unsupported routes revert to a prespecified
> fallback. Retrospective mechanism evidence across nine settings supports
> the core and conditional executors. A preregistered DS03 replay prospectively
> confirmed the fallback decision, but not predictive superiority over Engression.

## 11. 제출 판단

현재 패키지는 retrospective multi-domain evidence, 완전 동일 후보예산 baseline,
그리고 prospective route-selection evidence를 함께 갖는다. 다만 prospective
predictive superiority는 실패했으므로 “unseen cohort에서 최고 정확도”라고 쓰지
않는다. 최상위 저널에서 외부 일반화를 강화하려면 N-CMAPSS와 독립적인 실제
cohort 하나가 더 필요하다.
