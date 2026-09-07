# N-CMAPSS hard: PP와 일반 NN 비교

## 프로토콜

- 데이터: N-CMAPSS DS02-006
- test: 처음 보는 엔진 × 높은 TRA × 말기 구간, pooled 표본 수 159
- split 생성 seed: 42, 엔진당 최대 window 1,500
- PP 입력: 같은 causal window의 마지막 관측값, window 평균, 처음-끝 기울기
- PP 선택: train으로 학습하고 validation MSE로 9개 regularizer 후보 중 선택
- 공통 후처리 비교: 엔진별 RUL 단조성을 반영하는 unit isotonic
- 주 지표: test 159개를 합쳐 계산한 pooled R²

## 결과

| 모델 | seed | pooled R² 평균±표본 SD | RMSE 평균±표본 SD | 비고 |
|---|---:|---:|---:|---|
| cycle isotonic | 1 | 0.485 | 14.02 | 시간축만 사용 |
| LSTM+iso | 3 | 0.217±0.021 | 17.28±0.23 | raw sequence |
| sequence Transformer+iso | 3 | 0.767±0.006 | 9.42±0.11 | raw sequence, 15 epoch |
| TabPFN last-step | 1 | 0.934 | 5.02 | 보조 비교, 단일 seed·입력 형식 다름 |
| **PP raw** | **5** | **0.931±0.012** | **5.13±0.44** | PP 자체 출력 |
| **PP validation-adaptive multiscale** | **5** | **0.934±0.007** | **5.02±0.28** | causal feature preset을 validation에서 선택 |
| **PP+iso** | **5** | **0.886±0.004** | **6.58±0.12** | sequence NN과 같은 후처리 조건 |

seed 42–44만 맞춰도 PP+iso는 0.887±0.006이고 sequence Transformer+iso는 0.767±0.006이다. 차이는 pooled R² +0.120이다. 5-seed 예측 평균의 R²는 raw PP 0.934, validation-adaptive multiscale PP 0.937, PP+iso 0.887이다.

PP raw는 isotonic 후처리보다 높았다. 이 데이터에서는 네트워크가 이미 유효한 RUL 궤적을 만들었고, 사후 단조 보정이 개별 예측을 과도하게 평탄화했다. 따라서 PP 본 모델의 대표 성능은 raw 0.931±0.012로 보고하고, 기존 sequence benchmark와의 통제 비교에는 PP+iso 0.886±0.004를 함께 제시한다.

## 해석과 한계

이 결과에서는 PP가 일반 LSTM과 sequence Transformer보다 정확하고 seed 간 편차도 작다. PP의 affine tail과 latent regime gate가 높은 TRA의 말기 외삽에서 유효하다는 증거다. 다만 TabPFN 0.934는 단일 seed 보조 결과이고 PP raw 평균 0.931과 사실상 비슷하므로, 현재 결과만으로 TabPFN보다 우월하다고 주장하지 않는다.

추가 개발 실험에서는 causal window의 평균·표준편차·기울기를 여러 시간 규모로 만들고, feature preset과 separation regularizer를 validation MSE만으로 선택했다. 단일 seed 평균은 0.931±0.012에서 0.934±0.007로 개선됐고 ensemble은 0.934에서 0.937로 개선됐다. 따라서 TabPFN 단일 seed 0.934와 단일 모델 평균은 사실상 동률이고, seed ensemble은 PP가 +0.003 높다.

반면 validation 평균으로 `multiscale + separation=0.001` 하나를 모든 seed에 고정하면 0.923±0.008, ensemble 0.926으로 낮아졌다. 이는 multiscale feature 자체가 항상 우수한 것이 아니라 validation과 unseen-engine test 사이에 선택 불일치가 있음을 보여준다. 이 고정 설정은 최종 모델로 채택하지 않는다.

이 N-CMAPSS cohort는 이전 PAE 연구에서 이미 관측한 데이터다. 이번 PP 결과는 사후 개발 증거이며 새로운 untouched 검증이 아니다. 또한 Transformer는 raw sequence를, PP는 그 sequence의 causal summary를 받는다. 데이터 split과 test mask는 같지만 표현과 학습 예산이 완전히 동일한 architecture-only 대조는 아니다. 논문에는 이 제한을 명시하고, 최종 규칙을 고정한 뒤 별도의 unseen cohort에서 확인해야 한다.

PAE 결과는 이 비교에서 사용하지 않았다.
