# XJTU·FEMTO 구조 개선 실행 명세

## 목표와 현재 판단

목표는 올바른 입력과 같은 평가 조건에서 PP를 실제로 개선하는 것이다. 현재 기록으로 두 데이터셋의 개선 불가능성을 주장할 수 없다. 반대로 양수 R² 또는 0.3 달성도 보장하지 않는다. 이 문서는 구현할 후보와 이를 판정하는 실험을 지정한다. 새 학습 결과나 novelty 검증 완료를 뜻하지 않는다.

우선 FEMTO의 실제 PP 기준선을 복원하고 temporal residual을 검증한다. XJTU는 출력 범위 문제와 조건 이동 문제를 분리한다. 처음부터 대형 Transformer, latent scale, hazard, gate를 모두 합치지 않는다. 각 단계가 성공하면 다음 단계에 연결한다.

## 1. 반드시 알고 시작할 사실

| 기록 | 코드가 실제 수행한 작업 | 올바른 해석 |
|---|---|---|
| FEMTO −0.571 | 시간 metadata 열을 진동으로 사용 | 센서 모델 성능 근거에서 철회 |
| FEMTO −2.313 | 교정 입력, `direct_residual_mixture=True`, `fixed_affine_trust=0` | affine이 꺼진 PP 내부 NN 경로. 최종 modular PP의 한계가 아님 |
| FEMTO GRU −0.248 | 서로 다른 5개 GRU 예측을 평균 | ensemble 성능. 단일 GRU 평균은 −1.360, 표본 SD 0.819 |
| XJTU −0.843 | Ridge로 logit-progress 회귀, elapsed feature 추가 | neural PP와 동일하지 않으며 기존 입력 비교군과도 구분 필요 |
| lifetime/progress 후보의 큰 음수 | 지수 decoder와 frozen affine, 항상 양수인 support decay | 해당 구현의 실패. 모든 scale 모델의 실패를 입증하지 않음 |

기존 성공 9개 설정의 최종 모델은 `FINAL_PP_BENCHMARK_TABLE_KO.md`와 `final_modular_pp_evidence.py`의 실제 소스 연결로 찾는다. `evidence_gated_modules.py`의 HUST 0.724 등은 최종 모델 재현이 아니다.

## 2. 단계 A — 비교 기준과 구현 복구

### 공통

1. 입력 생성 함수와 라벨 생성 함수를 분리한다. prefix builder는 y, lives, 최종 recording 번호를 인자로 받지 않는다. 현재 사용 가능한 recording, 실제 경과시간, 센서, 운전조건만 받는다.
2. scaler는 해당 fold의 train에서만 fit. 표준화 전후 feature 범위·상수 열·NaN·overflow를 저장한다.
3. seed 42–44로 후보를 선택하고 42–46으로 최종 평가한다. 선택 점수는 unit별 RMSE를 동일 가중한 seed 평균. pooled RMSE도 기록한다. ensemble 점수만 최대화하지 않는다.
4. epoch 0를 checkpoint 후보에 포함한다. 후보마다 전체 search 결과, loss history, 학습 표본 수, 종료 epoch, 모델 checkpoint와 feature/split manifest를 저장한다.
5. 후보 선택을 마친 manifest를 저장한 후 test를 평가한다. 이미 본 test이므로 개발 결과로 표시한다. test 최고 후보를 validation 선택 결과로 바꿔 부르지 않는다.
6. `lifetime_scale.py`를 재사용하면 shape 비교의 chained inequality 수정, elapsed finite 확인, epoch당 permutation 1회 생성, epoch 0 평가를 먼저 고친다. 현재 batch마다 permutation을 새로 생성한다.

### FEMTO

- 소스: `experiments/femto_sensor_adapter_v2.py`. 6열 CSV의 4/5 가속도 열. Learning_set와 truncated Test_set만 사용. Full_Test_Set 금지.
- 공식 endpoint 11개를 주 평가로 유지한다. GRU와 PP는 정확히 같은 endpoint를 예측한다.
- 원본 split 재현: train 11,12,21,22,31 / val 32 / test 13,14,15,16,17,23,24,25,26,27,33.
- 비교군 B0: 동일 초기화·학습 조건의 direct MLP. B1: 실제 frozen affine + residual PP, `direct_residual_mixture=False`, affine trust 0 금지. B2: soft-anchor trainable affine PP. B3: 현재 32-step direct GRU.
- 기존 −2.313은 B1로 재명명하지 않는다. trust=0와 MLP가 동률이어야 하는 ablation에는 residual seed replay, zero-init, optimizer를 실제로 맞춘다.
- 현재 validation은 1개 bearing의 모든 행이고 test는 bearing당 endpoint 1행이다. 수정 프로토콜에서는 학습 bearing의 여러 prefix 절단점에서 마지막 시점만 채점하는 pseudo-endpoint validation을 추가한다.
- 원본 split 결과와 별도로 Learning bearing 6개 중 1개를 빼는 grouped CV를 수행한다. 각 holdout bearing의 관측 길이 25/50/75/90% 위치를 검증 endpoint로 쓴다. 백분위는 오프라인 validation 행 선택에만 쓰고 feature에는 넣지 않는다. 각 bearing의 기여를 같게 한다.
- CV 완료 뒤 6개 learning bearing 재학습은 별도 프로토콜이다. 비교군 모두 같은 6개를 사용하며 원본 5+1 실험과 별도 표로 보고한다. epoch는 CV 선택 epoch의 중앙값으로 고정한다.

### XJTU

- cache: `data/xjtu_sy/features_locked.npz`. 원본 split: train 37.5Hz11kN, val 35Hz12kN, test 40Hz10kN.
- train lifetime 최대 533 recordings, validation 최대 161, test 최대 2538. 기존 8-recording window에서 train 최대 RUL은 약 525이다. 실제 cache로 다시 확인한다.
- 원본 `fit_pp`/`predict`는 train 최대 RUL로 상한 clipping한다. 동일 학습 checkpoint에 capped/lower-only 출력을 먼저 비교하고, 다음으로 train·validation 모두 lower-only 계약인 모델을 별도 학습한다.
- test 라벨을 이용한 진단으로 `prediction=clip(y_test,0,train_max)`의 R²를 계산하면 상한 제한만으로 발생하는 최선 가능 오차를 정량화할 수 있다. 이는 oracle 진단이며 모델 성능으로 사용하지 않는다.
- train 운전조건이 하나여서 condition embedding만 추가해도 조건 효과를 배웠다고 할 수 없다. val 조건은 train과 반대 방향으로 이동한다. 조건 두 개로 재학습하려면 grouped CV를 두 조건 안에서 실시하고 별도 개발 프로토콜로 명시한다. 원본 split의 단순 버그 수정으로 부르지 않는다.
- elapsed는 windows 출력에 원본 `positions`를 직접 포함한다. 행 수로 재구성하지 않는다. unit ID는 split·loss 가중치에만 쓰며 prediction feature로 쓰지 않는다.

## 3. 1차 구조 — Multi-scale Temporal Residual PP (우선 구현)

### 입력과 encoder

관측 t에 대해 current feature x_t, 최근 8/32/128개 recording의 sequence, 실제 Δt, padding mask를 만든다. FEMTO는 교정된 14개 센서 통계와 운전조건을 사용한다. XJTU도 동일 정보 예산으로 시작한다. baseline 변화량은 그때까지 관측한 초기 prefix로만 계산한다. near-zero baseline으로 나눌 때는 train에서 정한 feature별 scale floor를 사용한다.

작은 causal TCN을 기본으로 한다: hidden 16 또는 32, kernel 3, dilation 1/2/4/8, right padding 없음, dropout 0 또는 0.1. 짧은 기록은 mask-aware pooling으로 처리한다. 같은 encoder를 사용하는 direct TCN을 필수 비교군으로 둔다. GRU를 동일 budget 보조 비교군으로 둔다.

### PP head

z_t = TemporalEncoder(history_t, mask_t, Δt)

y_hat = softplus((a_t + g_t b tanh(r(z_t)))/s) * s

a_t는 train current/causal feature에 적합한 affine 경로, r은 temporal residual, b는 train target scale에 대한 상대 bound, s는 작은 양의 smoothing scale이다. y_hat에 train 최대값 상한을 두지 않는다. softplus의 scale은 train에서 고정하고 PP/직접 NN 모두 같은 양수 decoder를 사용한다. 직접 ReLU를 대안으로 비교할 때 음수 영역의 gradient 소실을 기록한다.

초기 gate g=1로 시작한다. support decay를 적용하지 않은 후보가 반드시 있어야 한다. 이후 g=exp(-beta*d_state)에서 beta∈{0,0.1,0.5}만 비교한다. 운전조건 OOD와 센서 상태 OOD를 별도 기록한다. 상수 운전조건 축 하나 때문에 residual을 전부 끄는 설계를 기본값으로 쓰지 않는다.

loss = unit-balanced normalized RUL MSE + lambda_pair L_pair + lambda_anchor L_anchor.

L_pair = Huber(((y_hat_t - y_hat_u) - (time_u-time_t))/target_scale), 동일 unit의 실제 두 관측 t<u에 적용한다. run-to-failure label이 실제 시간 잔여량일 때만 유효하다. 재정비·재시작·RUL cap이 있는 경우 사용하지 않는다. lambda_pair∈{0,0.03,0.1}; L_anchor는 affine 이동을 제한하며 frozen/soft anchor를 비교한다. 이 penalty는 target-consistency 제약이며 단독 novelty라고 주장하지 않는다.

최소 ablation: direct temporal NN / current-feature PP / temporal PP / temporal PP+pair loss / temporal PP+pair+support gate. 표본이 작으므로 large model 수십 종을 무작정 검색하지 않는다.

### 이 모델을 먼저 하는 이유

현재 실패 실험은 temporal encoder가 없는 PP 또는 PP 경로가 없는 GRU였다. 따라서 기존 PP 구조와 history encoder를 공정하게 결합한 후보가 충분히 검증되지 않았다. raw waveform 모델보다 구현·비용이 작고 실패 지점을 분리할 수 있다. 성능 향상은 가설이며 GRU ensemble −0.248만으로 보장되지 않는다.

## 4. 2차 구조 — 상태·속도·열화 시작 분리 (1차 이후)

베어링은 정상 운전 시간이 길다가 열화가 시작될 수 있다. 모든 시점에서 전체 age를 진행률에 직접 대응시키면 정상 운전 길이가 다른 unit에서 큰 오차를 낼 가능성이 있다. 이를 검증하기 위한 후보다.

같은 encoder에서 degradation state q_t∈(0,1), positive speed v_t, onset score o_t를 출력한다. onset은 고장 유형의 확정 정답이 아니며 학습된 latent score로 명명한다. 관측된 변화량 예측 auxiliary task로 상태를 정규화하고, 동일 unit의 q가 완만히 증가하도록 soft pair loss를 준다. 경계를 단순 q=1로 선언하는 것만으로 실제 물리 경계가 생긴다고 주장하지 않는다.

직접 `(1-q)/v`만 사용하면 작은 v에서 폭발한다. denominator floor를 train-only scale로 정하고 saturation·floor hit 비율을 기록하며, raw-RUL loss를 반드시 포함한다. q와 v를 각각 식별했다고 주장하려면 auxiliary 관측/ablation이 필요하다. 이 후보가 direct temporal head보다 안정적으로 좋을 때만 PP에 채택한다.

Survival head는 3차 후보다. 연속시간 Weibull 등으로 survival S(u|history)를 정의하고 E[RUL]=integral S(u)du를 사용하면 fixed-bin horizon 상한을 피할 수 있다. 단, tail shape는 데이터 밖 가정이고 survival loss만으로 미관측 긴 수명 문제를 해결하지 않는다. prefix 수천 개는 독립 bearing 수천 개가 아니므로 unit-balanced likelihood가 필요하다. 먼저 temporal PP에서 잔여 오차 구조를 확인하고 결정한다.

## 5. raw signal/spectrum은 언제 추가하는가

1차 모델이 grouped validation에서 직접 temporal NN을 넘지 못하고 센서 요약량이 열화 변화를 분리하지 못하면, 64~128개 log-power frequency bin과 2-channel statistics를 결합한 작은 CNN encoder를 추가한다. 원신호 전체 Transformer부터 시작하지 않는다.

window별 진폭 정규화로 열화 RMS를 없애지 않는다. shape-normalized spectrum과 absolute amplitude를 별도 입력으로 보존한다. XJTU와 FEMTO의 sampling frequency는 각각 원자료 정의로 확인하고 frequency bin을 실제 Hz로 만든다. 필요 시 speed-normalized frequency를 추가하지만 이것이 unseen load response를 알아내는 것은 아니다.

self-supervised pretraining을 한다면 train bearing prefix에 한정한다. test history로 encoder를 업데이트하면 별도 transductive/TTA 실험이므로 원본 inductive PP 표에 합치지 않는다.

## 6. 결과 판정과 기존 PP 보호

- 주표: seed 5개 개별 pooled R², seed mean±sample SD, prediction ensemble pooled R², unit별 RMSE/MAE. 단일 모델과 ensemble을 동일 기준으로 비교한다.
- FEMTO endpoint는 unit당 1행이므로 unit R²를 만들지 않는다. 11 bearing 단위 paired bootstrap RMSE 차이로 불확실성을 표시한다.
- XJTU는 unit별 trajectory R²와 pooled R²를 모두 제공한다. bootstrap은 bearing 단위. 5개 test bearing으로 강한 유의성을 기대하지 않는다.
- 개선 기준: validation에서 선택된 신규 PP가 matched 기존 PP와 matched direct temporal NN보다 나은지 확인. R²>0 및 R²≥0.3 달성 여부는 별도 항목이다. 상대 개선만으로 성공이라 부르지 않는다.
- test 결과가 좋은 설정을 다시 선택하지 않는다. 다음 개발 라운드에 사용한 사실을 기록한다. 성공할 때까지 특정 test 최고값만 누적하는 표는 만들지 않는다.
- 기존 양수 9개 셋에서 새 구조를 실제로 같은 split에 적용해야 일반화/비열화를 판단할 수 있다. 기존 코드를 건드리지 않은 사실은 새 구조의 회귀검증이 아니다.
- 우선 HUST·NASA battery·MICH를 서로 다른 mechanism의 회귀 panel로 사용하고, 통과하면 최종 표 9개 전체를 평가한다. 각 셋 baseline은 최종 modular PP, 비교 metric은 같은 seed 통계다. 이 회귀 결과도 그대로 공개한다.
- 새 구조가 일부에서만 이기면 그 적용 범위를 보고한다. dataset name에 따라 test 승자를 고르는 router는 만들지 않는다. 기존 PP와 temporal PP 사이 선택이 필요하면 train/validation으로 고정한 규칙 자체를 비교한다.

## 7. 실행자가 만들 파일과 단계

1. `experiments/bearing_protocol_audit_v3.py`: 채널, 시간, label-free prefix, baseline floor, capped oracle 진단, split hashes 저장.
2. `src/pp_extrapolation/temporal_residual_pp.py`: mask-aware TCN + affine + signed bounded temporal residual + nonnegative uncapped decoder. train/val/test 공용 predict 경로.
3. `experiments/bearing_structural_v3.py`: `--dataset femto|xjtu --stage select|evaluate --protocol original|grouped_refit --output ...` 인터페이스 구현. 명령은 아직 존재하지 않으므로 파일 구현 후 실행한다.
4. `tests/test_temporal_residual_pp.py`: future deletion invariance, pad mask invariance, train support 밖 유한 출력, residual gradient/gate activity, seed replay 비교를 테스트한다.
5. `results/bearing_structural_v3/<dataset>/<protocol>/`: manifest, all_search.json, histories, checkpoints, row-aligned predictions, summary.json, report.md를 저장한다.
6. `STRUCTURAL_PP_REGRESSION_V3_KO.md`: 실제 최종 PP 대비 기존 성공 셋의 신규 구조 결과를 기록한다.

작업 순서: A 기준 복구 → FEMTO 1차 구조 → XJTU 상한 ablation 및 같은 1차 구조 → 기존 성공 셋 회귀 panel → 필요할 때만 2차 head 또는 spectrum encoder. 목표는 강건한 성능 향상이며, 각 라운드에서 무엇이 효과가 있었는지 설명할 수 있는 최소한의 비교를 유지한다.
