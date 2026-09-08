# MATR batch 2 temporal PP 개선 실험

## 결론

최종 PP의 affine·bounded residual 경로를 유지하고, 과거 8시점의 health·degradation rate·relative time을 읽는 causal GRU residual을 추가했다. 최종 PP와 **완전히 같은 train 11,552행, validation 673행, test 733행**을 사용했다.

| 모델 | test pooled R² | 판정 |
|---|---:|---|
| 최종 PP | **0.862** | 현 최종 모델 유지 |
| temporal PP, raw 5-seed ensemble | 0.661 | 기각 |
| temporal PP + validation-only transport, 5-seed ensemble | 0.860 | 기각: 최종 PP보다 0.0015 낮음 |
| temporal PP + validation-only transport, seed 42 | 0.914 | 가능성 확인용; 단일 seed 최고점을 최종 성능으로 채택하지 않음 |
| exact-row BatteryLife CPGRU ensemble | 0.537 | temporal PP보다 낮음 |
| V-REx | 0.850 | temporal PP ensemble과 유사 |

## 무엇을 확인했나

- 단순 Ridge affine을 쓴 v1의 낮은 성능은 temporal module 자체보다 base path가 약했던 것이 주원인이었다.
- 실제 최종 PP base를 사용하자 seed 42의 transported R²가 0.914까지 올랐다. 따라서 MATRb2에서는 과거 열화 궤적이 유효한 추가 정보다.
- seed별 transported R²는 `0.914, 0.859, 0.771, 0.801, 0.900`으로 분산이 크다. 평균 예측은 0.860으로 기존 PP 0.862를 넘지 못했다.
- Gaussian perturbation consistency loss를 validation에서 비교했지만, 선택된 계수는 0이었다. 이 형태의 consistency regularization은 채택하지 않는다.
- support decay 경계는 하드코딩하지 않고 해당 train sequence의 표준화된 최솟값에서 계산하도록 수정했다.

## 해석과 다음 모델

시간 이력을 넣는 방향은 맞지만, 현재 GRU residual은 작은 validation 차이에 따라 서로 다른 extrapolation correction을 학습한다. 다음 단계는 seed 평균을 쓰는 대신 train 내부 out-of-fold trajectory target으로 **단일 deterministic temporal head**를 학습하는 것이다. 이 head는 관측별 history 신뢰도와 support distance를 함께 출력하고, 신뢰도가 낮을 때 residual을 PP affine path로 수축해야 한다.

채택 조건은 다음과 같이 고정한다.

1. MATRb2 5-seed 또는 deterministic replay pooled R²가 0.862를 초과할 것.
2. HUST·Sunwoda·RWTH에서 기존 최종 PP보다 평균 성능이 떨어지지 않을 것.
3. MICH처럼 health–RUL mapping이 바뀌는 반례에서는 gate가 temporal correction을 거부하거나 실패 위험을 표시할 것.

현재 결과는 post-hoc development이며 새 untouched cohort의 확증 결과가 아니다.

재현 코드: `experiments/matr_batch2_temporal_pp_v2.py`, `experiments/matr_batch2_temporal_pp_v3.py`  
원시 결과: `results/matr_batch2_temporal_pp_v2/`, `results/matr_batch2_temporal_pp_v3/`
