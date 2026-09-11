# PP-X 상위저널 검증 패키지

작성자: 박진서  
논문 메인: **PP-X**  
방법 표현: **A validation-approved prior-residual framework for
contract-conditioned extrapolation**

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

전체 9개 setting을 하나의 30-candidate budget으로 다시 학습한 결과는 아직
없다. 그러므로 “모든 baseline과 완전히 동일한 탐색 예산”이라고 주장하지 않는다.

## 8. Prospective 감사

현재 저장소에 PP-X Paper Method v1을 검증할 **실행 가능한 미개봉 cohort는
0개**다.

- Alloy/MultiStage 및 sealed registry cohort: 이미 개봉
- Na-ion, XJTU, FEMTO, axial fan, Misata, MATWI 등: 이미 평가
- Oxford, SNL, Tongji, CALB, MEMSS 등: endpoint/eligibility를 열었거나 infeasible
- Perovskite: outcome 미개봉이지만 로컬 데이터·runner가 없고 기존 CCMR
  phenotype 계약이라 PP-X v1 confirmatory cohort로 바로 사용할 수 없음

따라서 prospective 결과를 만들었다고 보고하지 않는다. 다음 신규 cohort는
데이터 다운로드/endpoint 개봉 전에 현재 동결 프로토콜에 cohort ID, split,
adapter와 baseline budget을 추가 커밋해야 한다.

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
| 모든 모델 예산이 같은가 | MICH 확장 완료, 전체 9-setting equal-budget은 미완료 | 미완료 |
| prospective confirmation이 있나 | 없음으로 명시 | 미완료 |
| 하나의 실행 알고리즘인가 | frozen selector API·protocol·tests 구현 | selection 층 통과 |

## 10. 논문에서 사용할 결론

> PP-X does not claim universal accuracy. It formalizes extrapolation as the
> conditional execution of structural assumptions: an outcome-free contract
> restricts admissible priors, group-disjoint validation approves one
> prior-residual executor, and unsupported routes revert to a prespecified
> fallback. Retrospective mechanism evidence across nine settings supports
> the core and conditional executors, while prospective validity remains an
> explicit open requirement.

## 11. 제출 판단

현재 패키지는 강한 **retrospective methodology paper**로 정리할 수 있다.
그러나 최상위 저널에서 prospective generalization을 주장하려면 다음 두 항목이
추가로 필요하다.

1. 새로운 미개봉 cohort 1–2개
2. 가능하면 9개 main setting 전체의 equal-candidate-budget 재실행

이 둘이 없으면 제목·초록에서 “validated across unseen domains”보다
“retrospective multi-domain evidence”라고 정확히 제한한다.
