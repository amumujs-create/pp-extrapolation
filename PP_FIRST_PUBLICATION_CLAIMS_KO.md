# PP-X 선투고 권리·노벨티 배치

> **Canonical paper pointer:** 현재 선투고 대상과 paper-level 이름은 PP-X다.
> legacy PP/SAAR는 backbone·역사적 alias이고, CCMR v2.2는
> trajectory-domain executor evidence다. 최신 정의는
> `PPX_FINAL_PAPER_MODEL_KO.md`를 우선한다.

## 논문 순서

1. PP-X를 먼저 투고한다.
2. PP-X preprint/DOI를 확보한 뒤 PAE를 후속 작업으로 투고한다.
3. 동시 투고가 필요하면 PAE에서 PP-X preprint를 명시적으로 인용한다.

## PP-X가 소유하는 기여

- unit-disjoint, convex-hull-out strict-tail RUL 문제 정의
- outcome-free typed source contract와 admissible prior 제한
- frozen prior path + nonlinear residual common core
- validation physical-unit evidence로 executor를 승인하는 Algorithm 1
- boundary quotient, support-adaptive bound, history, regime transport executor
- direct/persistence fallback과 transferability-aware abstention
- PP-X와 plain NN, FT-Transformer, Engression, GP, Ridge 등 비교

## PAE가 나중에 소유할 기여

- PP-X가 다루지 않는 더 넓은 prior 후보의 자동 생성·검색
- 외부 지식에서 observation/domain contract 초안을 컴파일하는 상위 계층
- 여러 predictor family를 넘는 program-level coverage·regret 분석

PAE에서 typed contract, PP-X Algorithm 1, boundary multiplication, BQ-PP
network, validation approval와 fallback을 새로운 구조로 재주장하지 않는다.

## PP-X 초록의 핵심 표현

> Existing RUL predictors often fit in-support trajectories while leaving their
> behavior beyond the observed degradation support uncontrolled. We separate a
> stable extrapolative tail from bounded neural corrections and validate whether
> learned output errors can be transported across physical units. For domains with
> an observable failure boundary, a boundary-quotient extension guarantees zero
> RUL at failure while retaining a frozen affine quotient tail. Across multiple
> unit-disjoint strict-tail datasets, the framework improves pooled accuracy and
> exposes relationship-shift failures that ordinary random holdouts conceal.

## PP-X에서 강조할 것

- exact-zero gate의 최초성보다 frozen quotient과 bounded residual의 역할 분리
- ordinary future prediction과 다른 unit-disjoint strict support extrapolation
- MICH의 기존 PP `-1.522` 실패, bounded BQ-PP `0.468`, support-adaptive dual-scale PP `0.751`의 단계적 복구를 보고
- 성공 평균뿐 아니라 validation eligibility, fallback, failure taxonomy

## PAE와 공유하지 않을 주 결과

PP-X의 BQ executor 3-dataset 표를 PAE의 primary performance table로 복제하지
않는다. PAE는 상위 compiler가 PP-X 입력 contract를 어떻게 생성하는지를 별도
기여로 검증해야 한다.
