# FEMTO 고급 시계열 모델 필요성 진단

PP 공식 test를 확인한 뒤 수행한 retrospective 분석이다. 최근 16개 vibration summary를 causal sequence로 넣는 GRU direct regressor를 동일한 5개 Learning train bearing, 1개 validation bearing, 11개 공식 test endpoint에 적용했다.

| 모델 | pooled R² 평균±SD | 비고 |
|---|---:|---|
| PP | -1.378±0.000 | residual epoch 0 |
| Plain NN | -1.225±0.908 | 불안정 |
| GRU-16 | **-2.606±2.906** | 더 불안정하고 악화 |

GRU seed별 R²는 `-7.868, -0.914, -0.420, -3.636, -0.192`였다. 일부 seed는 PP보다 낫지만 모두 음수이며 초기화 분산이 매우 크다.

따라서 FEMTO 실패는 현재 증거상 “PP가 얕아서” 생긴 문제가 아니다. History encoder를 추가해도 bearing 6개의 작은 학습 표본과 약한 cross-bearing vibration–lifetime 관계를 해결하지 못했다. 이 데이터에서 더 큰 Transformer나 ensemble을 추가하면 분산과 과적합 가능성이 더 크다.

고급 모델이 필요해지는 조건은 더 많은 full run-to-failure bearing과 raw waveform self-supervised pretraining을 확보하는 경우다. 그런 데이터 없이 현재 PP 논문에 대형 sequence model을 넣는 것은 권하지 않는다.
