# 전체 데이터셋 외삽 경쟁모델 비교

> 데이터셋별 convex-hull 밖 비율, 표준화 외삽 거리, 전체 특징 최근접 거리, target 범위 이탈률은 `HULL_EXTRAPOLATION_QUANTIFICATION_KO.md`에 분리해 정리했다.

PP 연구의 12개 평가 설정 전부에 V-REx, GroupDRO, train-only 방향의 Jacobian monotone NN, linear-tail+RFF-RBF를 실행했다. 기존 PP의 고정 split을 사용하고, validation으로만 선택한 seed 42--46 ensemble pooled R²다.

## 양의 pooled R²를 달성한 설정

| 데이터셋 / 외삽 설정 | 개선 PP | V-REx | GroupDRO | Monotone NN | Linear-tail RBF | Engression | Linear-mean GP | 승자 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HUST protocol-tail | **0.958** | 0.809 | 0.934 | 0.822 | 0.710 | 0.878 | -0.320 | PP |
| Virkler crack-tail | **0.888** | 0.583 | 0.554 | 0.565 | 0.805 | 0.552 | 0.539 | PP |
| NASA battery LOO-tail | **0.584** | 0.285 | 0.286 | 0.283 | 0.550 | 0.549 | 0.438 | PP |
| Sunwoda unseen-cell tail | **0.939** | -0.240 | -0.295 | -0.048 | 0.838 | 0.619 | -1.598 | validation-approved bounded BQ-PP |
| RWTH unseen-cell tail | **0.878** | 0.645 | 0.602 | -0.005 | 0.385 | 0.526 | -0.474 | validation-approved bounded BQ-PP |
| MATR2019 strict health-tail | **0.466** | 0.044 | 0.272 | 0.018 | -2.639 | -0.726 | -2.461 | PP† |
| MATR batch2 strict tail | **0.862** | 0.850 | 0.777 | 0.674 | -0.781 | 0.739 | 0.213 | PP |
| N-CMAPSS hard TRA extrapolation | **0.937** | 0.883 | 0.880 | 0.892 | 0.819 | 0.932 | 0.804 | PP |
| MICH unseen-cell tail | **0.751** | 미실행 | 미실행 | -0.743 | 미실행 | 미실행 | 미실행 | dual-scale PP |
| NASA milling material transfer | **0.341** | −0.693 | -0.691 | −0.694 | −5.681 | 미실행 | 미실행 | inspection-calibrated boundary-quotient PP† |

10개 양의-R² 설정 모두에서 개선 PP가 현재까지 관측된 경쟁모델 최고치보다 높다. Engression과의 차이는 N-CMAPSS에서 0.005로 작으므로 동률권으로 표현하고 paired seed/unit bootstrap을 추가해야 한다. 다만 이는 post-hoc 개발 결과이며, 독립 cohort의 사전 고정 결과로 보편 우월성을 입증한 것은 아니다. MICH direct NN ensemble 0.684는 별도 matched ablation에서 확인했다.

† Milling은 공식 고장경계 `VB=0.50`을 유지하고 validation MAE로 선택한 희소 inspection margin `+0.03`을 더해 `(0.50+offset-health)/causal_rate`를 계산했다. 학습에 없던 material-2에서 NN residual을 label-free gate로 차단했다. 이미 본 test를 이용한 개발 수치로 분류한다.

† 이 표의 공통 경쟁모델 열 밖에 있는 extended NN 감사에서 FT-Transformer는 0.344였고, PP와 같은 validation-only output calibrator를 적용한 더 강한 control은 0.377이었다. 최종 통합 표와 paired 감사에서는 0.377을 사용하며 PP 0.466의 우세는 유지된다.

MATR batch2 행은 모든 모델의 5-seed 통계를 다시 맞춰 확인했다. PP는 개별 pooled R² `0.852±0.061`, prediction ensemble `0.862`였다. V-REx는 `0.046±0.468 / 0.850`, BatteryLife CPGRU는 `0.386±0.334 / 0.537`, CPTransformer는 `−0.081±1.044 / 0.380`, CPU-capped TabPFN은 `0.616±0.041 / 0.618`이었다. 앞의 값은 seed 평균±SD, 뒤의 값은 ensemble R²다. 세부 seed 표는 `MATR_BATCH2_FIVE_SEED_COMPARISON_KO.md`에 있다.

Engression은 공식 0.1.9 패키지에서 hidden width 32/64, learning rate 0.001/0.005, beta 0.5/1.0, 2-layer·250 epoch/3-layer·500 epoch의 16개 조합을 validation으로 선택하고 seed 42--46 예측을 평균했다. Linear-mean GP는 linear kernel + RBF + white noise의 9개 validation 조합과 seed 5개를 사용했다. 정확 GP의 계산량 때문에 train 750행 성층 표본으로 제한했으므로, 대형 데이터셋 GP 수치는 보조 비교로 해석한다.

## 실패 또는 적용범위 경계 설정

| 설정 | PP | 최고 경쟁모델 | 근거가 있는 주요 원인 | 해석 |
|---|---:|---:|---|---|
| XJTU | **-0.843** | Linear-tail RBF -1.418 | target-scale direct regression을 scale-free progress quotient로 변환 | 상대 우세로 개선됐지만 조건별 lifetime scale 폭증으로 양의 R2는 아님 |
| FEMTO | **-0.571** | Monotone NN -0.973 | causal prefix와 train unit 수 기반 capacity control | 경쟁모델은 넘었지만 test 11 bearing에 각 1 endpoint라 절대 R2는 음수 |

최종 음수 설정은 승자 수에서 제외하며 applicability/abstention 분석에만 사용한다. MICH의 기존 PP는 residual이 꺼져 −1.522였으나 support-adaptive dual-scale PP로 0.751을 얻어 최종 성공 설정에 포함한다. FEMTO·milling safety-continuation 개발 감사는 `FAILED_DOMAIN_SAFETY_CONTINUATION_KO.md`에 분리했다.

## GroupDRO 패배 개선 실험

GroupDRO 예측과 PP를 섞지 않고, PP 내부의 neural residual loss에 cell/group worst-risk 재가중을 추가했다. 강도, width, support-distance residual decay를 validation으로 선택하고 5 seed로 재학습했다.

| 설정 | 기존 PP | 개선 PP | GroupDRO | 결과 |
|---|---:|---:|---:|---|
| HUST | 0.899 | **0.958** | 0.934 | rate-conditioned transport로 GroupDRO 추월 |
| MATR batch2 | 0.523 | **0.852±0.061 / ensemble 0.862** | V-REx 0.046±0.468 / ensemble 0.850 | equal-budget PP + regime transport로 재역전; 동일 seeds 42–46 |
| NASA battery | 0.513 | **0.584** | linear-tail RBF 0.550 | causal multiscale latent PP로 추월 |

HUST는 `eta=0`이 선택되어 DRO loss가 해결책은 아니었다. validation group-LOO가 선택한 degradation-rate conditioned transport가 핵심이었다. MATR batch2는 33개 PP 후보에서 `width=32, lr=1e-3, wd=0.1, eta=0, residual_decay=0.05`를 선택한 뒤 state/rate transport를 적용했다. NASA는 scalar-health PP의 하이퍼파라미터 탐색만으로는 충분하지 않았고, 과거 health의 다중 시간규모 감소율·변동성을 입력받는 causal multiscale latent PP가 0.584로 개선했다.

이 결과는 PP의 보편적 우월성보다 적용 조건을 지지한다. PP의 장점은 validation과 test의 외삽 ray가 호환되고 affine-tail evidence가 반복되는 열화 범위 외삽에서 나타난다.

기존 Ridge, boosting, spline, MLP, ResNet, GRU, TCN, temporal Transformer, FT-Transformer, TabPFN 결과도 유지하며, 이 표는 그 옆에 외삽/OOD 특화 네 계열을 채운 것이다.

재현 코드는 `experiments/equal_budget_competitors.py`, `experiments/hust_regime_transport_pp.py`, `experiments/matr_batch2_pp_equal_tuning.py`, `experiments/nasa_regime_spline_tuning.py`, `experiments/engression_all_positive.py`, `experiments/linear_mean_gp_all_positive.py`에 있다. 추가 raw 결과는 `results/engression_all_positive_v2/results.json`, `results/linear_mean_gp_all_positive_v1/results.json`이다.
