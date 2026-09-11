# Tongji CY45-05 CCMR v2.0 결과

작성자: 박진서  
상태: **성능 성공** (`stable_bank`)

## 판정

사전 고정한 Tongji `CY45-05` 단일조건 고호트에서 CCMR v2.0이
validation·test 성공 기준을 모두 통과했다. PP-X v1에서 이미 zip을 연
적이 있으므로 dataset-level untouched는 아니고, CCMR progress 계약으로는
처음이다. counted sealed 자동 승격은 하지 않는다.

## 데이터

- 인구: `Tongji.zip` 중 filename에 `CY45-05` 포함 pkl 56개
- 적격 (≥50 cycle): 56/56
- train/val/test units: 33/11/12
- origins: train 구간 별도, validation 1185, test 959
- capacity CSV SHA-256:
  `f2d3dfb8e55178e75739a92f381901a79b9c0eff22f8f320cb83248c4fbf5912`

## Validation

- route: `stable_bank`
- expert: `experts_1_3`, deployment mass 0.82
- pooled R²: 0.999984
- pooled/macro RMSE 개선: 59.3% / 62.0%
- causal shadow coverage: (결과 JSON의 validation 필드)

## Test

- deployed coverage: 37.5%
- fallback replay error: 0
- Persistence pooled R²: 0.999965
- CCMR pooled R²: **0.999976**
- pooled/macro RMSE 개선: **18.0% / 30.4%**
- raw mean/CVaR/max regret: -43.7% / 0% / **0%**
- confirmatory success under frozen protocol: **true**

## 해석상 주의

cycle 단위 1-step이라 persistence R² 자체가 이미 0.9999대다. 절대
난이도는 MultiStage RPT(희소 점검)보다 낮다. 그래도 프로토콜 기준
(양의 R², ≥0.5% 개선, max regret 0%, coverage≥10%)은 충족했고,
unit 최악 손상 없이 persistence를 개선했다.

Engression 등 비교군 head-to-head는 이 문서 범위 밖이며, 필요하면
동일 sealed prediction에 retrospective로 붙인다.

## 산출물

- protocol: `protocols/TONGJI_CY45_05_CCMR_V20_PROTOCOL.md`
- extract: `experiments/extract_tongji_cy45_05.py`
- runner: `experiments/tongji_cy45_05_ccmr_v20.py`
- results: `results/tongji_cy45_05_ccmr_v20/results.json`
- predictions: `results/tongji_cy45_05_ccmr_v20/sealed_predictions.npz`
