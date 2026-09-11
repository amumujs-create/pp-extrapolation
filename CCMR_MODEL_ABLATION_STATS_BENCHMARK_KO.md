# CCMR 모형 ablation · 통계 · 벤치마크 요약

작성: 박진서  
산출: `results/ccmr_model_ablation_stats_benchmark_v1/results.json`  
그림: `ppt/pp/_build/figs/ccmr_*.png`

## 배포 / 거절

| 항목 | 상태 |
|---|---|
| 배포 | **CCMR v2.2** |
| 거절 | CCMR v2.3 AC-CRPE (엄격 GM RMSE 개선 실패) |

## 개발 ablation (v2.0 → v2.2)

- geometric RMSE ratio (v2.2/v2.0): **0.9986**
- mean relative RMSE gain: **+0.14%**
- strictly better: **1/5** (SIT LFP, small_crossfit_bank)
- non-worse: **5/5**
- false accept: **0**, max raw regret: **0%**
- paired sign-flip p: **1.0** (n=5)
- bootstrap 95% CI for RMSE gain: **[0, 0.42]%**

## Expert ablation (v2.0 bank)

단일 dynamics expert의 mean pooled gain은 음수 구간이 많다.  
프로토콜은 전역 expert 고정이 아니라 **bank selection / small-cohort route / exact fallback**이다.

## Holdout 벤치마크 (retrospective)

| 고호트 | 정확도 선두 | 안정성 (max regret 0) |
|---|---|---|
| Alloy A | Engression R²≈0.991 | CCMR v2.2 regret 0 (R²≈0.771) |
| MultiStage RPT | **CCMR v2.2** R²≈0.989 | **CCMR v2.2** (Engression max regret ≈205%) |

## v2.3 거절

- 최종 GM RMSE ratio vs v2.2: **1.000002** (≈0.0002% 악화)
- false accept 0, max regret 0, exact fallback OK
- 방화벽상 Alloy/MultiStage replay **미실행**
- 승격 **거부**

## 주장 경계

가능: v2.2 소표본 route 일반화·안정성; MultiStage 안정성 우위; v2.3 안전하지만 엄격 개선 실패로 거절.  
불가: Alloy에서 Engression 격파; v2.3 승격; Alloy 보고 재튜닝.
