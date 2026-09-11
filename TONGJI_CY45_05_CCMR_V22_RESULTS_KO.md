# Tongji CY45-05 CCMR v2.2 결과

작성자: 박진서  
상태: **성능 성공** (`stable_bank`)

## 판정

동일 Tongji `CY45-05` split/feature에 frozen CCMR **v2.2** 라우트를
적용했다. validation unit이 11개라 `small_crossfit_bank`가 아니라
`stable_bank`가 선택됐고, deployed 예측은 v2.0과 **완전 동일**하다.

## Test

- route: `stable_bank` (approved)
- coverage: 37.5%
- Persistence R²: 0.999965
- CCMR v2.2 R²: **0.999976**
- pooled/macro RMSE 개선: **18.0% / 30.4%**
- raw mean/CVaR/max regret: -43.7% / 0% / **0%**

## 비교군 (기존 ensemble 재점수, CCMR만 v2.2)

| 모델 | pooled R² | RMSE 개선 | max regret |
|---|---:|---:|---:|
| **CCMR v2.2** | **0.999976** | **18.00%** | **0%** |
| Engression | 0.999971 | 9.47% | 75.65% |
| Persistence | 0.999965 | 0% | 0% |

accuracy leader: CCMR v2.2. Engression 대비 RMSE·max regret 모두 우위.

## 산출물

- protocol: `protocols/TONGJI_CY45_05_CCMR_V22_PROTOCOL.md`
- runner: `experiments/tongji_cy45_05_ccmr_v22.py`
- results: `results/tongji_cy45_05_ccmr_v22/`
- competitors: `results/tongji_cy45_05_ccmr_v22_competitors/`
