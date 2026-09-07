# Convex-hull 외삽 정량화

## 공통 정의

- **Hull-out %**: 사전 선언한 PP 외삽 좌표 1차원에서 test가 train convex hull 밖에 있는 비율.
- **Hull distance**: train 평균과 표준편차로 표준화한 외삽 좌표의 hull 경계 밖 거리. 1.0은 train SD 1개에 해당한다.
- **Full-NN distance**: train 표준화 전체 특징공간에서 test와 가장 가까운 train row의 유클리드 거리.
- **Target-out %**: 진단용으로 test RUL이 train target 범위 밖인 비율. 이 수치는 모델 선택에 사용하지 않는다.

| 설정 | Hull-out % | median hull distance | max hull distance | median full-NN distance | Target-out % | 유형 |
|---|---:|---:|---:|---:|---:|---|
| HUST | 100.0 | 1.608 | 3.621 | 1.953 | 57.0 | 범위+프로토콜 외삽 |
| Virkler | 100.0 | 1.623 | 2.391 | 3.190 | 100.0 | 균열길이 범위 외삽 |
| NASA battery | 100.0 | 2.005 | fold 별 | 2.005 | 73.7 | health-tail LOO 외삽 |
| Sunwoda | 100.0 | 4.421 | 6.788 | 6.827 | 100.0 | 먼 unseen-cell tail |
| RWTH | 100.0 | 2.870 | 5.626 | 5.238 | 91.3 | unseen-cell tail |
| MICH | 100.0 | 3.426 | 5.060 | 5.929 | 86.1 | unseen-cell tail; shape 실패 |
| MATR2019 | 100.0 | 5.554 | 30.127 | 9.114 | 36.7 | 가장 먼 health-tail |
| MATR batch2 | 100.0 | 1.898 | 4.405 | 2.757 | 82.4 | health-tail |
| NASA milling | 100.0 | 1.508 | 3.334 | 6.689 | 100.0 | health + tool/unit shift |
| N-CMAPSS | 100.0 | 0.228 | 0.406 | 1.418 | 0.0 | 짧은 TRA hull 외삽, target범위 내 |
| XJTU | 0.3 | 0.000 | 4.332 | 운전조건 축에서 극단적 | 70.7 | 주로 condition/domain shift |
| FEMTO | 0.0 | 0.000 | 0.000 | 0.000 median | 0.0 | unseen-bearing endpoint shift |

XJTU와 FEMTO는 선언한 열화좌표의 convex-hull 외삽 데이터셋으로 분류하면 안 된다. 이 둘은 별도 domain-transfer stress test로 보고한다. N-CMAPSS는 hull-out 100%이지만 거리는 0.228 SD로 짧고 target 범위를 벗어나지 않으므로, 먼 RUL 범위 외삽보다 운전지표 경계 외삽으로 해석한다.

## 거리 shell 결과

기존 고정 PP 예측이 저장된 HUST, Virkler, NASA에서 test를 hull distance tercile로 나눈 결과다. shell 내 target 분산이 작으면 R²가 불안정하므로 RMSE를 주지표로 사용한다.

| 데이터 | near distance / RMSE | mid distance / RMSE | far distance / RMSE |
|---|---:|---:|---:|
| HUST | 0.486 / 100.34 | 1.608 / 51.81 | 2.866 / 52.32 |
| Virkler | 0.854 / 3.59 | 2.391 / 2.46 | 2.391 / 0.00* |
| NASA | 0.720 / 20.57 | 2.017 / 13.69 | 2.982 / 12.31 |

`*` Virkler far shell은 표본과 target 변동이 거의 없어 일반화 근거로 쓰지 않는다. 거리가 커질수록 오차가 단조 증가하지 않는 것은 먼 shell의 RUL 범위가 더 작은 것과 개체 구성이 다른 영향이다. 논문에서는 distance별 RMSE와 함께 shell 표본 수, target SD, unit 구성을 반드시 표기한다.

## 추가 경쟁모델 우선순위

1. **Engression**: 분포적 회귀로 support 밖 함수를 제약하는 직접 외삽 경쟁군. 스토캐스틱 학습이므로 현재 비교에서 가장 큰 공백이다.
2. **Progression**: 회귀 외삽 원리를 직접 표방하는 최근 모델. 코드와 적용 가정을 확인한 뒤 핵심 데이터셋에 적용한다.
3. **GPR with linear mean**: 작은 NASA/Virkler에서 유효한 고전적 불확실성 비교군. 대형 HUST/N-CMAPSS는 sparse GP가 필요하다.
4. **Density-Regression/DUQ**: point R² 경쟁보다 hull distance에 따른 불확실성·거부 성능 비교용이다.
5. **XGBoost/CatBoost, monotonic boosting**: 심사자가 예상하는 표준 기준선으로는 필요하지만, tree는 범위 밖 함수 외삽이 약하므로 논문의 핵심 외삽 모델로 보지 않는다.

생성 코드와 raw JSON은 `experiments/all_dataset_hull_audit.py`, `results/all_dataset_hull_audit_v1/results.json`에 있다.
