# PP 선투고 권리·노벨티 배치

## 논문 순서

1. PP를 먼저 투고한다.
2. PP preprint/DOI를 확보한 뒤 PAE를 후속 작업으로 투고한다.
3. 동시 투고가 필요하면 PAE에서 PP preprint를 명시적으로 인용한다.

## PP가 소유하는 기여

- unit-disjoint, convex-hull-out strict-tail RUL 문제 정의
- frozen affine tail + bounded nonlinear residual PP backbone
- RUL/health-margin 공간의 Boundary-Quotient PP
- group-LOO·seed consensus를 이용한 output-error transport
- identity fallback과 applicability abstention
- PP, BQ-PP, plain NN, FT-Transformer, Engression, GP, Ridge 비교

## PAE가 나중에 소유할 기여

- typed observation/domain contract
- admissible prior compilation
- prior strength selection과 prior-off routing
- PP를 포함한 executor 선택의 coverage·regret·failure analysis

PAE에서 boundary multiplication, BQ-PP network, PP calibration을 새로운 구조로
재주장하지 않는다.

## PP 초록의 핵심 표현

> Existing RUL predictors often fit in-support trajectories while leaving their
> behavior beyond the observed degradation support uncontrolled. We separate a
> stable extrapolative tail from bounded neural corrections and validate whether
> learned output errors can be transported across physical units. For domains with
> an observable failure boundary, a boundary-quotient extension guarantees zero
> RUL at failure while retaining a frozen affine quotient tail. Across multiple
> unit-disjoint strict-tail datasets, the framework improves pooled accuracy and
> exposes relationship-shift failures that ordinary random holdouts conceal.

## PP에서 강조할 것

- exact-zero gate의 최초성보다 frozen quotient과 bounded residual의 역할 분리
- ordinary future prediction과 다른 unit-disjoint strict support extrapolation
- MICH의 `-1.522 -> 0.468` 복구와 unit 31 실패를 동시 보고
- 성공 평균뿐 아니라 validation eligibility, fallback, failure taxonomy

## PAE와 공유하지 않을 주 결과

PP의 BQ-PP 3-dataset 표를 PAE의 primary performance table로 복제하지 않는다.
PAE는 compiler가 PP executor를 올바르게 승인·거절하는지를 주 결과로 사용한다.
