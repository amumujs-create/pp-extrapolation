# 현재 최종모형 — CCMR v2.2

작성자: 박진서

## 배포 상태

- **현재 배포 모형:** CCMR v2.2 small-cohort crossfit route
- **동결 manifest:** `protocols/CCMR_V22_FROZEN_MANIFEST.json`
- **예측기:** CCMR v2.0 causal dynamics bank를 그대로 사용
- **v2.2 변경:** validation physical unit이 3–9개인 경우
  `small_crossfit_bank` route를 추가
- **후속 v2.3:** 엄격 평균 RMSE 개선 실패로 거절, 배포하지 않음

## 구조

1. 네 개 causal dynamics expert를 physical-unit crossfit으로 적합한다.
2. validation에서 mean/CVaR20/max raw regret 계약을 만족하는 correction
   mass와 expert 조합만 허용한다.
3. fold consensus와 context support 안에서만 correction을 활성화한다.
4. validation unit 수와 causal-shadow 증거에 따라
   `stable_bank`, `small_crossfit_bank`, `cautious_causal`,
   `exact_fallback` 중 하나를 선택한다.
5. 증거가 부족하면 persistence를 정확히 복원한다.

## 동결 개발 결과

- 개발 도메인: 5
- false accept: 0
- maximum test raw regret: 0
- exact fallback: 전 도메인 일치
- v2.0 대비 strict win: 1 (SIT LFP)
- v2.2/v2.0 geometric RMSE ratio: 약 0.9986

## 검증 경계

- 동일 split 경쟁모형은 일부 도메인에서 더 높은 정확도를 보이지만
  worst-unit regret가 크게 증가했다.
- v2.2의 5-domain 성능 우월은 통계적으로 유의하지 않다
  (domain sign-flip p=1.0).
- 정당한 핵심 주장은 universal SOTA가 아니라
  **risk-aware selective correction**이다.
- Alloy A와 MultiStage RPT는 열린 holdout이며 구조 재튜닝에 사용하지 않는다.
- 신규 미개봉 prospective cohort 확증은 아직 남아 있다.

## 재현

```bash
PYTHONPATH=src python experiments/ccmr_v22_trajectory_development.py
PYTHONPATH=src python experiments/ccmr_v22_paper_structure_ablation.py
PYTHONPATH=src python experiments/ccmr_v22_validation_gap_audit.py
PYTHONPATH=src python experiments/ccmr_v22_matched_development_benchmark.py
```

기존 결과 디렉터리가 있으면 각 실험은 덮어쓰기를 거부한다.
