# Multi-Stage Stage-2 TP_z CCMR v2.0 결과

작성자: 박진서  
상태: **성능 성공 아님** (stable_bank 배포 후 test 기준 미달)

## 위치

Stage-2 나머지(`TP_z*`)다. Stage-1 봉인 고호트와 **동일 물리 cell ID
72/72**가 겹치므로 독립 확증이 아니다. same-cell continued /
selection-conditioned replication으로만 기록한다.

## 추출

- `experiments/extract_multistage_stage2_tpz.py`
- 72 cell, 999 RPT 행, 유한 999
- CSV SHA-256 `b7910af94b8d92bb3548b516a49a37ca57a758fdd4e9151dd56efac0834185c6`

## 적격·split

- eligible (≥10 RPT): 60
- train/val/test units: 36/12/12
- validation/test origins: 36/29

## Validation

- route: `stable_bank`
- expert: `experts_0_1`, deployment mass 1.0
- pooled/macro RMSE 개선: 44.1% / 49.7%
- pooled R²: 0.948
- raw mean/CVaR/max regret: -72.4% / -39.0% / -6.0%

## Test (봉인 후)

- deployed coverage: 24.1%
- fallback replay error: 0
- Persistence pooled R²: **-0.344**
- CCMR pooled R²: **-0.342**
- pooled/macro RMSE 개선: **0.076% / 2.23%**
- raw mean/CVaR/max regret: -19.7% / 0% / **0%**

성공 기준 중 pooled R²>0 및 pooled 개선 ≥0.5%를 동시에 만족하지
못해 `performance_success=false`다. max regret 0%만으로는 성능 성공으로
세지 않는다.

## 해석

긴 RPT 궤적(스크리닝 ≥10이 60 cell)이라 적합은 가능했지만, Stage-2
후반 progress 구간에서 persistence 자체 R²가 음수라 외삽 난이도가
Stage-1 성공 레짐과 다르다. validation에서 크게 이긴 보정이 test
평균 정확도로 거의 이어지지 않았다.

독립 성공 고호트나 “비교군을 이긴 새 확증”으로 쓰지 않는다. 비교군
승기 문장은 기존 Stage-1 + CCMR v2.0 retrospective 결과를 유지한다.

## 산출물

- protocol: `protocols/MULTISTAGE_STAGE2_TPZ_CCMR_V20_PROTOCOL.md`
- results: `results/multistage_stage2_tpz_ccmr_v20/results.json`
- predictions: `results/multistage_stage2_tpz_ccmr_v20/sealed_predictions.npz`
