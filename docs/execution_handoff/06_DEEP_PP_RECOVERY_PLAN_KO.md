# PP 추가 개선: 코드 재감사와 실행 설계

## 결론

다음 실행자는 큰 모델부터 추가하지 말고 (1) decoder/초기화 정합성, (2) 손실과 검증의 단위 일치, (3) causal temporal 표현을 순서대로 수정한다. 그 뒤 학습 가능한 열화 진행 속도를 가진 PP head를 검증한다. 현재 0.257은 보존할 개발 결과지만 일반화된 구조 개선이 완성됐다는 증거는 아니다. 이 문서는 설계와 읽기 전용 재계산 결과이며 신규 학습 결과가 아니다.

## 1. XJTU 0.257을 정확히 해석하기

Raw progress-temporal PP ensemble R²는 −0.187이다. 0.257은 validation scale을 반대 방향으로 반사한 뒤 얻은 결과다. 마지막 계산에서 test y를 사용하지 않는다는 사실과, 보정법 선택이 test와 독립이라는 주장은 다르다. 대화에서는 test global-scale sweep의 양수 개선을 본 뒤 `2-a_val`을 도입했다. 따라서 이 구조는 명백한 post-test 개발이다.

`c_train=0,c_val=-1,c_test=1`이라는 입력 기하는 맞지만 출력 오차도 선형·대칭이라는 결론은 따라오지 않는다. `a_train=1`도 out-of-fold calibration으로 확인하지 않고 둔 anchor다. 조건 변화와 bearing 개체 차이가 섞여 있으므로 해당 반사 규칙을 모든 데이터의 공용 prior로 만들면 안 된다.

저장된 5-seed 평균 예측을 다시 계산한 결과:

| test unit | raw RMSE | reflected RMSE | raw unit R² | reflected unit R² |
|---|---:|---:|---:|---:|
| Bearing3_1 | 877.3 | 557.5 | −0.442 | 0.418 |
| Bearing3_2 | 887.9 | 574.2 | −0.527 | 0.361 |
| Bearing3_3 | 503.7 | 940.1 | −21.982 | −79.038 |
| Bearing3_4 | 464.3 | 716.3 | −0.138 | −1.707 |
| Bearing3_5 | 137.9 | 264.1 | −18.923 | −72.096 |

Unit-equal mean MSE는 409,260→421,402로 약 3% 악화했다. 긴 두 bearing이 전체 6,999행 중 5,020행을 차지하므로 pooled R² 개선과 unit-level 안정성 악화가 동시에 가능하다. 사용자의 주 지표인 pooled R²를 계속 사용하되 unit RMSE와 함께 보고한다. 목적은 pooled R² 상승과 짧은 bearing 붕괴 완화다.

## 2. 지금까지의 구조 실험에 남은 구현 문제

### 2.1 Progress decoder의 의미

`experiments/xjtu_progress_temporal_pp_v3.py`에서 raw score f에 softplus를 적용하고 RUL=t*expm1(softplus(f))로 복원한다. latent clip이 비활성인 영역에서는 항등식에 따라 이것은 **RUL=t*exp(f)**다. f가 시간에 따라 충분히 감소하지 않으면 RUL이 t에 비례해 커진다. 이는 버그라기보다 강한 귀납 가정이며, total-life 또는 monotone progress가 자동 보장되지 않는다.

또 Ridge는 z=log1p(y/t)를 target으로 fit하지만 계수를 softplus 전 raw score에 복사한다. r=0에서 Ridge 예측을 보존하지 않는다. 예를 들어 z=0은 decoder 이후 RUL=t로 변한다. NN residual이 ±0.5로 제한돼 이 불일치를 충분히 보정하지 못할 수 있다. 먼저 inverse-softplus target 또는 다른 decoder로 초기화를 맞춰야 한다.

### 2.2 TCN의 causal/mask 검증

`src/pp_extrapolation/temporal_residual_pp.py`의 GroupNorm(1,C)은 temporal 축까지 통계를 모은다. 이것을 각 시간점의 causal encoder라고 부르면 틀리다. 단, endpoint마다 허용된 prefix만 넣은 최종 예측이 바로 test 미래를 본다는 뜻은 아니다. 시간별 상태 재사용이나 padding 길이 변화에서는 문제가 된다.

Padding을 마지막 pooling에서만 제외하며 convolution bias와 normalization에는 들어간다. 기존 padding test는 PP residual이 zero-init이라 encoder 변화가 출력에 드러나지 않을 수 있다. residual을 nonzero로 만든 테스트와 direct mode 테스트가 필요하다.

### 2.3 반복된 실험의 근거 한계

- `xjtu_two_condition_refit_v2.py`는 10개 중 지정한 2개 unit을 holdout하여 8개로 학습한다. 10개 전체 최종 refit도, 전체 grouped CV도 아니다. 따라서 모든 refit 전략이 실패했다고 결론 내리지 않는다.
- 과거 unit-scale calibration 진단은 test unit 전체 궤적 median을 사용했다. 과거 t 시점에는 미래 recording이므로 배포 가능한 calibration이 아니다. 그 실패로 causal calibration 전체를 배제할 수도 없다.
- Spectrum 실험은 채널당 32 power + 16 envelope bin을 GRU에 붙인 것이다. 학습 가능한 raw-waveform CNN이나 self-supervised encoder를 검증한 것이 아니다. 그 계열 전체를 실패로 분류하지 않는다.
- 5개 seed의 낮은 표준편차는 최적화 안정성이다. 표본이 독립 bearing 5개인 일반화 불확실성과 다르다.
- FEMTO −2.313은 affine trust=0 경로다. 이를 실제 최종 affine-residual PP의 한계로 삼지 않는다.

## 3. 첫 실행: decoder를 바꾼 PP의 matched ablation

원본 split과 같은 feature/행으로 다음을 비교한다. 기존 0.257 모델은 그대로 보존한다.

**D0 원본:** t*exp(f), 기존 초기화 그대로 재현.

**D1 정합 초기화:** z=log1p(y/t), raw_target=inverse_softplus(max(z,epsilon)). inverse_softplus(z)=z+log(-expm1(-z))를 사용한다. epsilon 후보는 개발셋에서 고정하고 endpoint y=0 근처의 민감도를 보고한다. affine는 raw_target에 fit한다. epoch 0가 해당 affine decoder와 일치하는지 테스트한다.

**D2 additive log-RUL PP (우선):** train에서 정한 양의 s에 대해 y=s*softplus(f). raw_target=inverse_softplus(max(y/s,epsilon)). t를 출력 앞에 곱하지 않으며 elapsed는 관측 feature로만 준다. 현재의 시간 비례 편향을 제거하는 ablation이다. affine+bounded residual 구조는 유지한다. direct NN도 동일 decoder를 쓴다.

**D3 log1p-RUL PP:** z=log1p(y/s), RUL=s*max(expm1(z_hat),0). affine는 z에 fit하고 residual은 signed로 학습한다. 음수 z의 decoder/gradient 처리는 명시한다. 지수 overflow 방어와 clip-hit 비율을 저장하며 train 최대 RUL로 clipping하지 않는다.

D1–D3를 한 번에 복잡한 temporal 모델에 넣지 않는다. 먼저 동일 current-feature MLP residual에서 decoder 효과를 확인하고, 유효한 decoder를 기존 GRU residual로 옮긴다. 이렇게 해야 좋아진 원인이 출력 계약인지 encoder인지 알 수 있다.

### 손실

목적함수는 latent MSE 하나로 끝내지 않는다. L=lambda_raw*normalized raw-RUL MSE + lambda_log*Huber(log1p(y_hat/s)-log1p(y/s)) + lambda_pair*L_pair.

Raw scale s는 train에서만 계산하고 train loss는 unit 균형 가중치를 적용한다. lambda_raw는 1로 고정, lambda_log∈{0,0.1,1}, lambda_pair∈{0,0.03,0.1}. 상대 손실 때문에 짧은 RUL의 작은 분모가 폭발하지 않도록 직접 y로 나누는 MAPE는 쓰지 않는다.

L_pair는 같은 bearing의 t<u에서 Huber(((y_hat_t-y_hat_u)-(time_u-time_t))/s)다. 실제 run-to-failure RUL일 때만 사용한다. 불확실성이 줄면서 예측이 수정될 수 있으므로 hard monotone constraint로 강제하지 않는다. clipping된 label 또는 정비 후 restart에는 적용하지 않는다.

검증 선택은 개발셋 pooled RMSE를 주 지표로 하되 unit RMSE 평균과 최악 unit RMSE를 함께 저장한다. validation OOF에서 단일 unit에 과도한 손해를 주는 후보는 보조 제약으로 거절할 수 있다. 거절 기준은 test 평가 전에 고정한다.

## 4. 두 번째 실행: 적응형 residual 용량을 가진 temporal PP

현재 bad affine를 작은 bound로 유지하는 것은 안정적 실패를 만든다. frozen affine + bounded NN 철학을 유지하되 bound를 한 숫자로 고정하지 않는다.

z_t=GRU(H_t, Δt, mask), f_t=a(x_t)+b_t*tanh(r(z_t)).

b_t=b_small+(b_large-b_small)*sigmoid(g(z_t)). b_small/b_large는 train target 변환 scale에 맞춰 개발셋 후보로 정한다. gate가 학습 중 correction capacity를 늘릴 수 있지만 곧바로 불확실성 확률이라고 부르지 않는다. 과도한 b_t를 억제하는 작은 penalty와 gate saturation 로그가 필요하다.

이미 구현된 final dual-scale BQ-PP에서 residual capacity 관련 아이디어와 학습 규칙을 우선 확인한다. 기존에 없는 새 novelty라고 중복 주장하지 않는다. RUL bearing용 적용에서 실제로 효과가 있는지 검증한다.

최소 비교: 같은 encoder direct NN / frozen affine-only / fixed-bound PP / adaptive-bound PP / adaptive-bound+pair loss. 기여가 없는 module은 제거한다. support gate는 beta=0 후보를 반드시 포함하고, 상수 condition 축의 OOD만으로 residual을 끄지 않는다.

Encoder는 우선 기존 16-wide GRU로 고정한다. TCN을 쓸 경우 timewise LayerNorm 또는 norm 없는 causal block, 각 block의 mask 처리를 구현한다. 논문 실험 전에 future-deletion, padding-length, nonzero-residual tests를 통과해야 한다.

## 5. 세 번째 실행: 관측으로 보정되는 열화 진행 clock

앞 단계가 실패하면 total-life를 매 관측에서 임의로 재예측하는 대신 상태와 진행 속도를 모델링한다. 이것은 고급 후보이며 먼저 구현할 기본 모델이 아니다.

q_t는 latent damage state, v_t=softplus(v_head(z_t))는 진행 속도다. 예측 상태 q^-_{t+1}=q_t+v_t*Δt에 실제 센서 관측으로 계산한 innovation correction을 더한다. 상태 재구성 또는 다음 관측 예측 auxiliary loss가 있어야 한다. y만으로 q와 v를 분해하면 무한히 많은 조합이 가능해 분리했다고 주장할 수 없다.

run-to-failure 학습 unit의 종료 상태를 q≈1로 정규화할 수 있다. 이는 알려진 물리 고장경계가 아니라 학습된 latent 경계다. 미래 v를 일정하게 가정한 head와 neural evolution head를 ablation한다. (1-q)/v를 사용할 때 train-only v floor, floor hit rate, 예측 발산을 기록한다. 낮은 v의 장수명 구간이 부족하면 extrapolation 가정이 결과를 지배한다.

강점 가설: 동일 진동 세기라도 빠르게 악화 중인지 장시간 안정적인지 구별한다. 약점: train unit 수가 작고 고장 형태가 다르면 미래 regime를 알 수 없다. 새 이름이나 물리 단어를 붙이는 것만으로 해결되지 않는다.

## 6. FEMTO의 별도 우선순위

FEMTO는 XJTU 반사 scale을 이식할 근거가 없다. 먼저 6개 Learning bearing의 leave-one-bearing-out validation을 만들어 prefix 25/50/75/90%에서 마지막 시점만 채점한다. 비율은 오프라인 검증 위치 선택용이며 prediction feature로 넣지 않는다. 원본 5+1 프로토콜 결과는 별도로 유지한다.

원본 32-step GRU가 가진 seed 불안정성과 validation 1개 unit 선택 불안정을 먼저 줄인다. 전체 refit epoch는 여러 fold의 선택 epoch 중앙값으로 고정한다. 정규화는 fold train에서만 fit한다. refit할 때 모든 비교군에 같은 learning unit을 제공한다.

그 후 위 D2 decoder와 adaptive-bound PP를 같은 GRU에서 평가한다. 아직 실패하면 spectrum의 absolute amplitude와 normalized shape를 분리한 작은 CNN을 검증한다. 특정 bearing geometry/fault frequency는 실제 출처가 없으면 추측해서 넣지 않는다.

단순 32+16 spectrum bin GRU 실패는 raw CNN 실패가 아니지만, CNN이 좋아질 것이라는 보장도 없다. 같은 representation의 direct NN보다 PP가 좋은지를 반드시 확인한다.

## 7. 실험 예산과 실행 순서

1. 기존 저장 예측 재계산, decoder identity/초기화·mask 테스트, 현재 row fingerprint 생성.
2. XJTU D0–D3, train/validation 선택, 3 screening seeds. 모든 decoder에 동일 encoder·시간 예산. test 비교는 선택 완료 후 수행.
3. 선택 decoder의 adaptive-bound/pair loss 비교. 상위 2개만 5-seed 최종 학습. hyperparameter·checkpoint를 test 최고점으로 교체하지 않는다.
4. FEMTO grouped validation에서도 동일 학습 API를 실행한다. 데이터셋별로 사용할 수 있는 prior만 adapter에서 선언한다.
5. HUST/NASA battery/MICH에서 실제 최종 PP와 회귀 panel 비교. 단순 기본 PP를 baseline으로 착각하지 않는다. 신규 모델이 개선되면 9개 성공 설정 전체로 확대한다.
6. 위 단계가 통과하지 않을 때만 clock/state model 또는 raw CNN을 개발한다. untouched cohort는 기존 규칙이 고정될 때까지 보류한다.

기존 0.257 reflection은 ablation으로 남기고, 새 main model 성능은 reflection 없이도 보고한다. 모든 comparator에 동일 validation-only calibration을 제공한 비교를 추가한다. 모델 고유 이득과 calibration 효과를 분리하지 못하면 PP 구조 기여로 주장하지 않는다.

## 8. 구현 파일 계약

- `src/pp_extrapolation/rul_decoders.py`: encode_target/decode_prediction/init_raw_target, round-trip tests, unit metadata와 stable transforms.
- `src/pp_extrapolation/adaptive_temporal_pp.py`: encoder/current affine/residual bound head, 공통 positive decoder, optional pair loss. `forward_components`로 affine/residual/bound/prediction 로그 제공.
- `experiments/bearing_recovery_v4.py`: select/evaluate/refit 단계 분리, 원본/그룹CV protocol 구분, 모델·feature·seed·row hash manifest.
- `experiments/bearing_recovery_v4_audit.py`: seed/ensemble, unit RMSE, early/mid/late 오차, output range, gate saturation, correction size, time consistency 계산.
- `results/bearing_recovery_v4/`: all_search, selected_config, training curves, checkpoints, predictions(unit,position,y,pred,seed), paired unit bootstrap report.

위 파일은 이 문서 작성 시 아직 신규 구현하지 않았다. 이름만 보고 존재하는 실행 파일로 가정하지 않는다. 원본 test label 파일은 evaluator만 읽도록 구성한다. 과거 개발 test라는 사실은 별개로 문서에 유지한다.

## 9. 채택 기준과 중단 기준

개선 목표는 XJTU pooled R² 0.257 초과와 unit-equal RMSE 개선을 동시에 확인하는 것이다. 0.3은 희망 목표이지 사전 성능 약속이 아니다. 단일 seed 평균과 ensemble 모두 보고한다. 5개 seed가 모두 양수여도 unit 일반화가 증명되는 것은 아니다.

FEMTO는 같은 encoder 직접 NN보다 개선되는 것과 pooled 양수 달성을 구분한다. 비교군만 낮추어 PP 우세를 만들지 않는다. 기존 좋은 9개 설정은 신규 구조를 실제로 적용한 뒤에만 유지/악화를 판정한다.

이 순서를 한 번 수행하고도 효과가 없으면 제한된 domain-transfer stress test로 남길 수 있다. 이미 맞는 prior를 가진 9개 설정의 PP 논문 기여를 별도 정리한다. 양수 수치를 만들기 위해 split, EOL label, 평가 endpoint를 조용히 바꾸지 않는다.
