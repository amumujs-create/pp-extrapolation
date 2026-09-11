# 현재 논문 메인 모델 — PP-X

작성자: 박진서

## 명칭과 범위

- **논문 메인 모델:** PP-X
- **정식 설명:** contract-conditioned prior-residual architecture with
  transferability-aware abstention
- **CCMR v2.2:** PP-X의 trajectory-domain risk-certified executor이자
  구조·안전성 검증 사례
- **CCMR v2.3:** 개발 엄격 개선 실패로 거절된 후보

PP-X는 CCMR 하나와 동의어가 아니다. PP-X가 상위 학습·실행 프레임워크이고,
CCMR은 causal trajectory에서 expert bank, raw-unit risk certificate,
small-cohort route와 exact persistence fallback을 구현한다.

## 논문 메인 구조

1. train 전에 domain contract와 허용 prior family를 선언한다.
2. frozen extrapolative prior 주위에 nonlinear residual을 학습한다.
3. validation/source OOF 증거로 executor와 prior 사용 여부를 결정한다.
4. 전이 증거가 부족하면 direct/neural 또는 persistence safety 경로로 후퇴한다.
5. test label이나 test-batch 통계로 route를 바꾸지 않는다.

## CCMR이 제공하는 PP-X 구조 증거

- bank를 상시 켜면 개발 도메인 max raw-unit regret가 30.3%까지 증가한다.
- v2.2 small-crossfit route를 제거하면 SIT LFP 이득이 사라진다.
- 단일 linear-rate expert는 RADAR에서 2% max-regret cap을 위반한다.
- frozen v2.2는 개발 5도메인에서 false accept 0, max regret 0을 유지한다.

이 결과는 **PP-X의 선택적 prior 사용과 안전 후퇴가 필요한 이유**를
보여 주는 mechanism evidence다. PP-X 전체가 모든 도메인에서 정확도
1위라는 뜻은 아니다.

## 논문 주장 경계

가능:
- PP-X는 데이터 계약에 따라 prior/executor를 선택하고 근거가 부족하면
  abstain/fallback하는 외삽 프레임워크다.
- CCMR ablation은 risk-aware route와 fallback의 구조적 필요성을 지지한다.

불가:
- PP-X가 모든 OOD/RUL 데이터에서 보편적 SOTA다.
- CCMR v2.2의 5-domain 평균 성능 향상이 통계적으로 유의하다.
- Alloy/MultiStage를 보고 PP-X 또는 CCMR 구조를 재튜닝했다.
- 신규 미개봉 prospective 확증이 끝났다.

## 기준 문서

- 논문용 모델 정의: `PPX_FINAL_PAPER_MODEL_KO.md`
- 전체 개발 프레임워크: `PPX_FINAL_MODEL_KO.md`
- CCMR 구조 ablation: `CCMR_V22_PAPER_STRUCTURE_ABLATION_KO.md`
- CCMR 검증 보강: `CCMR_V22_VALIDATION_GAPS_RESOLVED_KO.md`
- CCMR 동결 manifest: `protocols/CCMR_V22_FROZEN_MANIFEST.json`
