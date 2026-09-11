# LED 3-cohort 강한 외삽 결과

작성자: 박진서  
평가일: 2026-09-10

## 평가 계약

- 신규 소용량 cohort: Data_ID_2, 5, 6
- 총 파일 크기: **246,685 bytes**
- 독립 physical unit: 각각 162, 134, 50
- train/validation/test physical unit 완전 분리
- 2-step 미래 health 예측
- train forecast origin: 수명 진행도 0–30%
- validation: 40–60%
- test: 75–100%
- 세 cohort 모두 test 행 100%가 train 진행도 밖
- 실제 시간 기준 train 최대 밖 비율: 100%, 91.9%, 100%

따라서 같은 시간 구간의 임의 holdout이 아니라 unseen unit의 깊은 미래구간
예측이다.

## 기존 absolute-output v1.4

기존 모델은 baseline보다 RMSE는 줄였지만 절대 calibration이 붕괴했다.

- Data_ID_2 R²: -175.07
- Data_ID_5 R²: -229.42
- Data_ID_6 R²: -51.17

초기 health가 약 1인데 깊은 외삽에서 평균 0.80–0.83을 예측했다. age feature의
선형/신경 경로가 작은 실제 열화량보다 훨씬 큰 하강을 외삽한 것이 원인이다.
따라서 `overall_success=false`다.

## OAIR v1.5

후속 구조는 **Persistence-Anchored Ordered-Axis Invariant Residual**이다.

`future health = current health + gamma * bounded residual(slopes)`

핵심은 다음과 같다.

1. exact baseline은 현재 health를 그대로 유지하는 persistence다.
2. residual 입력에서 health level과 age를 제거해 외삽축 자체를 다시
   외삽하지 않는다.
3. slope 1/3/4만으로 작은 미래 변화량을 추정한다.
4. train 변화량 95백분위수의 두 배로 correction을 제한한다.
5. validation raw unit-risk budget으로 residual mass를 정한다.
6. deployment에서는 그 mass의 25%만 사용한다.
7. slope support 밖에서는 exact persistence로 복귀한다.

이 구조는 v1.4 점수를 본 뒤 만든 보정이므로 v1.5 결과는 개발 증거다.

## 결과

### Data_ID_2

- persistence: R² 0.5420, RMSE 0.008429
- OAIR: **R² 0.5451, RMSE 0.008401**
- mean/CVaR/max regret: -0.44% / +0.45% / +0.91%
- OOD fallback: 5/79행

### Data_ID_5

- persistence: R² 0.5037, RMSE 0.008109
- OAIR: **R² 0.5154, RMSE 0.008013**
- mean/CVaR/max regret: -1.11% / +1.43% / +2.55%
- OOD fallback: 2/123행

### Data_ID_6

- persistence: R² 0.2759, RMSE 0.021694
- OAIR: **R² 0.2763, RMSE 0.021689**
- mean/CVaR/max regret: +0.02% / +0.18% / +0.18%
- OOD fallback: 2/20행

세 cohort 모두:

- 양의 test R²
- persistence 대비 RMSE 비악화
- raw mean/CVaR/max 2%/5%/10% cap 통과
- strong-origin test 100%

따라서 사전 정의한 개발 성공 기준은 `overall_success=true`다.

## 해석

강한 외삽에서 중요한 것은 더 복잡한 absolute predictor가 아니었다. 변화가
작은 시스템에서는 현재 상태를 anchor로 고정하고, ordered axis와 독립적인
작은 residual만 허용하는 구조가 안정적이었다.

다만 개선 폭은 Data_ID_2와 6에서 작다. 이 결과로 보편적 우월성이나 독립
확증을 주장할 수 없다. 다음 미개봉 자료에서는 v1.5 구조와 threshold를
그대로 고정한 뒤 한 번만 평가해야 한다.

## 재현

```bash
python experiments/led_strong_extrapolation_v14.py
python experiments/oair_v15_led_strong_development.py
```

결과:

- `results/led_strong_extrapolation_v14/results.json`
- `results/oair_v15_led_strong_development/results.json`
