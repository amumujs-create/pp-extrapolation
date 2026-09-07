# MATR 외삽 경쟁모델 추가 비교

고정된 MATR2019 셀 분할과 strict health-tail row를 그대로 사용했다. 모든 강도와 구조는 validation cell에서만 선택했고, 선택 후 seed 42--46으로 다시 학습했다. 평가지표는 pooled R²이다.

| 계열 | 모델 | pooled ensemble R² | single-seed mean ± SD | RMSE | macro R² |
|---|---|---:|---:|---:|---:|
| 제안법 | **최종 PP (원본 PP + dual-evidence affine transport)** | **0.466** | 별도 원본 프로토콜 | **54.76** | 0.012 |
| 일반 NN | FT-Transformer | 0.344 | 0.303 ± 0.037 | 60.66 | -0.145 |
| 환경 강건 학습 | **GroupDRO-MLP** | **0.272** | -0.196 ± 0.434 | 63.94 | 0.004 |
| 시계열 NN | GRU | 0.332 | 0.171 ± 0.189 | 61.23 | 0.174 |
| 일반 NN | MLP | 0.206 | -0.162 ± 0.459 | 66.75 | -0.147 |
| 위험 외삽 | V-REx MLP | 0.044 | -0.361 ± 0.245 | 73.23 | -0.003 |
| 구조 제약 | Jacobian monotone MLP | 0.018 | -0.381 ± 0.246 | 74.23 | -0.010 |
| 추세+국소 커널 | linear-tail + RFF-RBF | -2.639 | -2.959 ± 0.417 | 142.93 | -3.978 |

PP 0.466은 원본 300-epoch latent PP의 validation-only output transport 결과다. 나머지 확장 NN 표는 공통 150/200-epoch 후속 벤치마크이므로, 정확한 동일 예산 순위라고 과장하지 않고 모델 계열별 스트레스 테스트로 사용한다. 원본 예산에서 직접 비교된 수치는 PP 0.257, FT 0.331이며, PP의 transport 승인 규칙까지 포함한 최종 executor가 0.466이다.

## 추가 모델의 의미

- **V-REx**: train cell별 위험 분산을 벌점으로 줘 관측 환경 밖 위험을 외삽한다. 단순 ERM과 다른 domain-generalization 비교군이다.
- **GroupDRO**: train cell 중 현재 손실이 큰 환경에 가중치를 올린다. 새 비교군 중 가장 강했지만 PP보다 pooled R²가 0.194 낮았다.
- **Jacobian monotone MLP**: 건강도에 대한 RUL 미분이 음수가 되면 벌점을 준다. 단조 prior 하나만으로는 MATR의 셀별 regime 차이를 해결하지 못했다.
- **linear-tail + RFF-RBF**: support 밖에서는 RBF 성분이 사라지고 선형 평균이 남는 고전적 외삽 기준선이다. validation에서 선택했어도 크게 실패했다.

PP와 GroupDRO의 cell별 MSE를 paired bootstrap하면 PP가 10개 중 7개 cell에서 우세했고 평균 MSE 차이(PP-GroupDRO)는 -772.6이었다. 다만 95% cell-bootstrap CI는 [-1640.6, 160.1]로 0을 포함한다. 따라서 이 한 cohort만으로 PP가 GroupDRO보다 통계적으로 확실히 우월하다고 쓰지 않고, pooled 성능 우세와 cell 이질성을 함께 보고한다.

이번 결과는 모든 외삽 모델을 포괄하지 않는다. EV는 모델이 아니라 외삽 validation 절차이고, Engression은 분포적 외삽 가정과 stochastic response가 필요한 별도 계열이라 현재 deterministic RUL 입력과 동일 예산 비교로 억지 구현하지 않는다. 논문 본문에는 Ridge/boosting/spline, MLP/ResNet/GRU/TCN/Transformer/TabPFN, V-REx/GroupDRO, monotone NN, linear-tail kernel을 합친 표를 싣는 구성이 적절하다.

코드와 원시 결과는 `experiments/extrapolation_competitors_matr.py` 및 `results/extrapolation_competitors_matr_v1/`에 있다.
