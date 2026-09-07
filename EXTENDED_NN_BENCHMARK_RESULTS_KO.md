# 추가 NN 벤치마크 결과

MATR 2019 동일 row 분할. 사후 비교이며 모델 종류 전체에 대한 최적 성능을 의미하지 않는다. 9개 후보를 seed 42 validation으로 선택한 뒤 5개 seed 재학습. 150 epoch / patience 25. 이전 300 epoch / seed별 선택 결과와 프로토콜이 다르므로 직접적인 개선·열화 원인으로 단정하지 않는다.

| 모델 | pooled ensemble R² | single mean | seed SD | RMSE | macro R² |
|---|---:|---:|---:|---:|---:|
| ft_transformer | 0.344 | 0.303 | 0.037 | 60.664 | -0.145 |
| gru | 0.332 | 0.171 | 0.189 | 61.226 | 0.174 |
| temporal_transformer | 0.220 | -0.054 | 0.420 | 66.146 | -0.210 |
| mlp | 0.206 | -0.162 | 0.459 | 66.750 | -0.147 |
| tabpfn | 0.202 | 0.087 | 0.352 | 66.935 | -0.940 |
| unconstrained | 0.171 | -0.103 | 0.502 | 68.205 | -0.546 |
| moe | 0.112 | -0.459 | 0.800 | 70.600 | -0.016 |
| fixed | 0.069 | -0.032 | 0.091 | 72.277 | -0.816 |
| pp | -0.100 | -0.631 | 0.952 | 78.586 | -0.201 |
| single | -0.391 | -1.150 | 1.399 | 88.362 | -0.468 |
| boosting | -0.416 | — | — | 89.149 | -1.939 |
| pp_joint | -0.688 | -1.374 | 1.087 | 97.336 | -1.181 |
| no_affine | -0.706 | -2.101 | 0.829 | 97.847 | -0.614 |
| tcn | -1.393 | -1.570 | 0.122 | 115.898 | -2.992 |
| resnet | -1.764 | -2.167 | 0.617 | 124.558 | -3.497 |
| ridge | -2.145 | — | — | 132.869 | -2.517 |
| spline | -31.614 | — | — | 427.851 | -62.857 |

기존 확증 latent PP 0.257은 원본 프로토콜의 결과로 별도 보존한다. 이번 pp는 regularizer를 끈 구조 대조이며 pp_joint는 중간 결과 확인 이후 추가한 탐색군이다. Test에 따른 모델 선택을 새로운 확증 성공으로 보고하지 않는다.

TabPFN은 CPU 1,000 train rows, 1 internal estimator를 사용한 보조 조건이다. GRU/TCN/temporal Transformer는 같은 과거 8개 관측을 직접 받고, 표형 모델은 그 관측으로 만든 6개 요약 입력을 받는다.

RTDL ResNet/FT-Transformer는 rtdl-revisiting-models 0.0.2 공식 패키지. TCN은 causal dilated CNN task adaptation. https://github.com/yandex-research/rtdl-revisiting-models

raw/clipped row predictions, validation 후보, refit checkpoints는 results/extended_nn_benchmark_v1 에 저장한다. 체크포인트는 state_dict이며 모델 재구성에는 코드와 train 자료가 함께 필요하다.

## 외삽 전용·OOD 경쟁군 추가

| 모델 | pooled ensemble R² | 비고 |
|---|---:|---|
| 최종 PP + dual-evidence transport | **0.466** | 원본 PP 프로토콜 |
| GroupDRO-MLP | 0.272 | cell을 environment로 학습 |
| V-REx MLP | 0.044 | cell risk variance 벌점 |
| Jacobian monotone MLP | 0.018 | health-RUL 방향 제약 |
| linear-tail + RFF-RBF | -2.639 | 선형 외삽 평균+국소 커널 |

상세 설계와 통계 비교는 `EXTRAPOLATION_COMPETITOR_RESULTS_KO.md`에 있다.

## 제출 관점의 결론

- 표형 FT-Transformer 0.344, 과거 window GRU 0.332가 기존 latent PP 확증 점수 0.257보다 높았다. GRU unit-macro R²도 0.174로 양수다. 현재 결과로 PP가 강한 NN보다 우수하다고 주장할 수 없다.
- 동일 공통 탐색 프로토콜의 PP 구조 대조는 -0.100, 추가 PP joint 탐색은 -0.688이었다. 추가 튜닝이 항상 개선되지 않았고 validation에서 선택된 설정의 test 이전이 불안정했다. 다만 기존 PP와 regularizer, epoch, seed 선택 규칙이 달라 한 요인의 인과 효과로 해석할 수 없다.
- 기존 고정 gate/단일 expert 비교와 이번 비교의 순위가 달라 gate 효과도 학습 설정에 의존한다. 구조적 신규성은 아직 강하게 증명되지 않았다.
- 이 결과는 한 cohort의 사후 벤치마크다. FT/GRU 대비 우월성이나 불확실성/선택적 예측의 이득을 주장하려면 개발 cohort에서 방법을 고정한 뒤 독립 평가가 필요하다.
- 학습량은 모델당 9개 후보 + 5개 refit, 최대 150 epoch라는 제한된 예산이다. 모든 종류의 NN 또는 각 모델의 최적값을 시험한 것이 아니다. 기존 PP regularizer 탐색을 동일한 300 epoch 예산으로 완전히 교차 탐색하지는 않았다.

검증: 13개 NN 실험군 × 5 seed = 65개 저장된 row prediction에서 ensemble 지표를 재계산해 JSON과 일치함을 확인했다. 기존 테스트 34개 및 모델별 forward/backward smoke check도 통과했다.
