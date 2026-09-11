# 신규 소용량 고호트 PP-X v1.4 결과

작성자: 박진서  
평가일: 2026-09-10

## 결론

신규 자료 두 종류를 열었지만 엄격한 확증 계약은 모두 적격성에서 중단됐다.
후속 LED 탐색 실험에서 기존 v1.4는 같은 시험군 내부에서는 개선했으나,
새 시험조건의 시간척도 이동에서는 prior를 잘못 켜 baseline보다 악화했다.

따라서 현재 결론은 **자동 레짐 clustering만으로는 강건성이 충분하지 않으며,
train-support 밖에서는 prior를 정확히 0으로 만드는 OOD guard가 필수**라는
것이다.

## 1. NASA ALT 2023

- 원본 ZIP: 513 MB
- 26 pack, expanded telemetry 약 3.4 GB
- reference-discharge capacity만 추출한 요약: **41,279 bytes, 475행**
- 사전 EOL: 정규화 capacity가 두 번 연속 80% 이하
- 적격 pack: 6/26
- 고정 split 적격: train 3, validation 2, test 1
- 사전 최소 test 4팩을 충족하지 못해 모델을 학습하지 않고 inconclusive

결과:
`results/nasa_alt_small_summary_v14_screen/results.json`

## 2. 외부 LED 6-set

- 공개 원본 CSV 여섯 개 총 **1,971,430 bytes**
- source commit:
  `07edea554ab0e157719b24ffc10dddfa43945db0`
- 사전 L70 test set Data_ID_2/6의 적격 unit: 0
- 따라서 L70 확증도 모델 점수 없이 inconclusive

L70 실패 뒤 endpoint 범위를 보고 L92.5를 고른 별도 실험은
**post-screen exploratory development**이며 확증이 아니다.

### L92.5 내부 test

- Data_ID_1 physical unit key: `(LED_type, Sample_ID)`
- 적격 13 unit, train/validation/test 7/3/3
- strict-tail rows 65/10/8
- baseline: R² 0.5113, RMSE 736.33
- auto-regime v1.4: **R² 0.6267, RMSE 643.52**
- mean/CVaR/max regret: -9.74% / +0.32% / +0.32%

같은 시험군 안에서는 trust .40, alpha .25 route가 유효했다.

### Data_ID_4 시험조건 이동 stress

- 적격 12 unit, strict-tail 68행
- baseline: R² -5952.14, RMSE 3241.93
- auto-regime v1.4: **R² -7293.86, RMSE 3588.71**
- mean/CVaR/max regret: **+11.91% / +18.86% / +22.05%**

v1.4는 이 set의 대부분을 validation에서 유리했던 레짐으로 분류해 prior를
켰다. 실제로는 RUL 시간 단위와 열화 속도 척도가 달라 prior와 baseline이 모두
절대 실패했고, prior는 손상을 더 키웠다.

## 원인 증거

train support distance:

- validation: 평균 1.75, 최대 2.91
- 같은 시험군 internal test: 평균 2.02, 최대 2.87
- 시험조건 이동 stress: 평균 118.84, 최소 41.65

즉 stress 68행 전부가 validation support 최대값보다 멀었다. K-means는 항상
가장 가까운 기존 cluster를 반환하므로, 절대적으로 먼 신규 레짐도 높은
confidence의 기존 레짐처럼 보인 것이 실패 원인이다.

## Post-test OOD guard ablation

실패를 본 뒤, support distance가 validation 최대 2.91을 넘으면 prior mass를
0으로 두는 진단 guard를 추가했다. 이는 사후 보정이므로 원래 v1.4 결과를
대체하지 않는다.

- internal test: 0/8행 차단, R² 0.6267 유지
- stress: 68/68행 차단, exact baseline 복원
- stress regret: mean/CVaR/max 모두 0
- combined regret: -0.19% / +0.004% / +0.013%

추가 prior 손상은 막았지만 stress baseline R² 자체가 -5952이므로 절대 성능은
해결되지 않았다. 이 영역은 같은 direct-continuation law가 아니라 시간척도
transport expert가 필요하다.

## 판정

1. 기존 v1.4는 신규 시험조건 이동에서 강건성 기준을 실패했다.
2. v1.4의 레짐 posterior에는 반드시 **absolute support rejection**이 필요하다.
3. 다음 버전은 `known-regime prior bank + unknown-regime baseline/transport`
   구조로 사전 등록해야 한다.
4. OOD guard와 transport expert는 새로운 미개봉 set에서 다시 평가해야 하며,
   이번 stress 결과를 확증으로 재분류하지 않는다.

후속 strong-origin forecasting 결과와 persistence-anchored OAIR v1.5 보정은
`LED_STRONG_EXTRAPOLATION_RESULTS_KO.md`에 별도로 기록했다. 세 LED cohort
모두 양의 R²와 risk cap을 달성했지만 post-test 개발이라는 한계는 유지된다.

재현:

```bash
python experiments/nasa_alt_extract_small_summary.py
python experiments/led_l925_v14_development.py
```
