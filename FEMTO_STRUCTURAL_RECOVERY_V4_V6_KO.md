# FEMTO 실제 구조 개선 실험: v4–v6

2026-09-09. **아직 해결하지 못했다. 신규 후보는 채택하지 않는다.**

수정된 acceleration columns 4/5 캐시를 사용했다. 기존 결과/모델은 덮어쓰지 않았고 새 cohort는 사용하지 않았다. 모든 결과는 이미 본 FEMTO test에서의 개발 결과다.

## 실행 결과

각 값은 5개 학습 seed(42–46) 예측 평균의 official truncated endpoint pooled R²다. test는 11개 bearing당 마지막 허용 관측 한 행이며 공식 RUL 초 단위다.

| 실험 | 학습/검증 | PP R² | matched NN R² | 판단 |
|---|---|---:|---:|---|
| v4: causal multiscale + 고차원 affine + bounded residual | Learning 6개 grouped CV → 전체 6개 refit | −4.141 | −9.688 | 둘 다 실패 |
| v5: 조건/age 4차원 affine + 같은 residual, soft-anchor 후보 | 위와 동일 | −4.428 | −9.688 | 저차원 prior도 해결 못함 |
| v6: 저차원 affine + GRU16 residual | 기존 5개 train + Bearing3_2 validation | −2.898 | −0.701 | 동일 encoder에서 PP가 더 나쁨 |

이전 교정 GRU 결과 −0.248은 다른 초기화/출력 cap 구현을 사용한 별도 기준이다. v6가 그 값을 재현한 것은 아니다. v6 두 비교군은 모두 lower-only 출력을 사용한다. v4/v5와 기존 5+1 결과는 학습 unit과 검증 방식이 달라 구조 효과로 직접 빼면 안 된다.

v6 PP seed별 R²: −2.945, −2.827, −3.192, −2.714, −3.103. ensemble만 나쁜 것이 아니다. v6 direct: −0.654, −0.747, −1.179, −2.044, −3.485.

## 무엇을 바꾸었나

v4/v5는 causal signed-log sensor history, 4/16/64 관측 horizon의 평균 변화·표준편차·실제 시간 기울기·관측량을 사용한다. 초반 history의 기준값은 당시 사용 가능한 최초 최대 10개 관측으로만 계산한다. frozen affine를 inverse-softplus target에 fit하고 softplus(affine+bounded NN)로 출력한다. y=0은 epsilon 근사다. train-only target scale과 unit 균형 loss를 사용한다.

v4는 affine에 모든 history 특징을 넣고, v5는 초기 affine를 condition one-hot과 log age 4개에 제한했다. v5는 frozen/soft-anchored 두 가지를 검증했고 frozen이 선택됐다. soft-anchor 후보는 이후 전체 affine 계수 조정을 허용한다.

v6는 실제 순서 정보를 사용하는 32-step GRU16이다. prior는 condition/log-age에 대해 raw normalized RUL을 Ridge로 학습한 frozen affine이며, NN residual bound .25/1/3을 validation에서 비교했다. 3이 선택됐다. 동일 GRU direct를 함께 학습했다. 이것은 기성 NN의 예측을 앙상블한 모델이 아니라 내부 affine+bounded residual이다.

## 실패 원인에서 확인된 것과 아직 가설인 것

**확인:** v4 validation CV RMSE는 PP 약 5774초, direct 약 3611초였다. v5 PP는 약 4522초로 validation에서는 개선했지만 test는 개선하지 않았다. v6 validation에서도 PP가 direct보다 나빴다. 따라서 test만 유독 나빠진 문제가 아니라 source 검증에서부터 prior가 유용하지 않은 경우가 있었다.

**확인:** 같은 운전조건에서도 Learning 수명이 크게 다르다. condition1: 28020초/8700초, condition2: 9100초/7960초, condition3: 5140초/16360초다. condition/age만으로 개체별 수명 차이를 충분히 설명할 수 없다는 가설에 부합한다. 이 수치만으로 진동 정보의 식별가능성이 없다고 증명할 수는 없다.

**확인:** v4/v5의 NN도 크게 실패했다. 따라서 모든 실패를 PP의 bounded residual 탓으로 돌릴 수 없다. 요약 특징, 작은 unit 수, fold마다 다른 최적 epoch와 전체 refit의 불일치도 분리해야 한다.

**한계:** v4/v5는 fold별 최저 validation loss로 후보를 고르고 epoch 중앙값으로 전체 refit했다. fold별 최적 epoch가 0–180으로 크게 달랐고 일부는 상한에 닿았다. 이 결과는 모든 grouped selection 전략의 실패를 의미하지 않는다. 다음 재실험에서는 공통 epoch의 평균 fold loss와 3개 selection seed를 사용해야 한다. 이번 screening은 seed42이며 최종 모델만 5 seed다. '충분한 대규모 튜닝 완료'라고 부르지 않는다.

## 다음에 시도할 가치가 있는 구조

현재의 14개 요약 sensor 특징에 대한 affine+residual 반복보다, 진동 주파수 성분 변화와 bearing 고유 baseline을 구별하는 표현을 먼저 검증할 필요가 있다. 기존 고정 spectrum bin 시험은 raw waveform CNN 시험과 같지 않다.

권장 다음 단계는 학습 bearing의 waveform crop 두 개로 train-only representation을 학습하고, causal sequence encoder의 direct head 대 PP head를 동일 조건에서 비교하는 것이다. PP는 현재 절대 진동 크기 prior 대신 학습된 상태와 상태 변화율에 제한적인 correction을 가한다. 이는 아직 구현/검증하지 않은 후속 가설이며 성공이나 양수 R²를 보장하지 않는다. test waveform을 self-supervised fit에 포함하면 transductive 별도 프로토콜로 표시해야 한다.

## 검증 및 산출물

- 미래 sensor를 변경해도 이전 시점 multiscale feature가 불변인지 통과.
- zero residual에서 PP가 초기 affine 경로를 그대로 복원하는지 통과.
- nonzero GRU residual이 설정 bound를 넘지 않는지 통과.
- Python compile 및 git diff whitespace 검사 통과.
- v4/v5: `experiments/femto_grouped_structural_v4.py`, 후자는 `--lowdim` 옵션.
- v6: `experiments/femto_affine_gru_recovery_v6.py`.
- `results/femto_grouped_structural_v4/`, `results/femto_grouped_structural_v5/`, `results/femto_affine_gru_recovery_v6/`에 selection manifest, 학습/검증 곡선, seed별 endpoint 예측, 결과 JSON 저장.
- v4/v5는 train loss와 validation curve, v6는 validation curve를 저장한다. test 결과로 최종 표의 모델을 교체하지 않았다.
