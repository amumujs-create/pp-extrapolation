# FEMTO 우선 복구: 원시 진동 구조 실험 결과

2026-09-09. **미해결. 기존 최종 모델로 채택하지 않는다.** 신규 구조 세 가지를 실제 실행했고 각각 encoder부터 5개 독립 seed로 학습했다. 각 seed 안에서는 PP/direct가 같은 frozen encoder를 사용한다. 새 cohort, Full_Test_Set, test label 학습은 사용하지 않았다.

## 결과

공식 Test_set 11개 bearing endpoint, 초 단위 RUL, 5개 예측의 평균에 대한 pooled R².

| 구조 | PP | 같은 encoder의 direct GRU | frozen latent-affine only |
|---|---:|---:|---:|
| v7 raw CNN + frozen latent-affine + causal GRU residual | −1.358 | −1.126 | −1.458 |
| v8 causal 초기 진동 기준 + 관측 종료 지점 validation | −1.347 | **0.075** | −1.432 |
| v9 진행률 학습 + elapsed 기반 RUL 복원 | −52.193 | −29.444 | −51.496 |

후속 구조도 실행했다.

| 후속 구조 | pooled R² | 결론 |
|---|---:|---|
| v10 condition별 health-threshold 도달시간 | −12.485 | 공통 진동 임계점 가정 실패 |
| v11 같은 조건 Learning 궤적 정렬·RUL transport | −11.485 | 비슷한 진동 궤적이 비슷한 RUL을 뜻하지 않음 |
| v12 Learning 6개 전체 raw-CNN refit | −2.332 | validation bearing 추가 학습만으로 해결되지 않음 |
| v13 latent failure-mode survival PP | −1.864 | direct total-life GRU −2.544보다 높지만 mode=1 선택, 절대 성능 실패 |

v8 direct의 양수는 **PP의 성공이 아니다.** direct 개별 seed R²는 −.795, −.868, −1.178, −1.311, −1.618이다. 평균 예측에서만 양수가 됐으며 단일 재학습 배포가 안정화됐다고 주장할 수 없다. 기존 corrected GRU −.248보다 평균 예측값은 높지만 v8은 validation 프로토콜도 바뀌어 구조 단독의 이득으로 해석하지 않는다.

v7 PP는 직전 같은 5+1 split의 저차원 affine+GRU PP −2.898보다 수치상 개선했지만, 양수도 아니고 matched NN도 이기지 못했다. 기존 최고 성능 해결로 보고하지 않는다.

## 실제 구현

- acceleration columns 4/5의 2560×2 파형을 그대로 읽었다. 4323개 기록의 float32 cache는 약 84.4 MiB다. 고정 FFT bin을 다시 CNN이라고 부른 것이 아니다.
- 파형을 기록별 RMS로 나누어 모양을 입력하고 log RMS는 별도 context로 보존한다.
- Conv1D(2→8→16→16), 평균/최대 pooling, 24차원 latent를 train RUL supervision으로 학습했다. self-supervised pretraining은 이번에 구현하지 않았다.
- encoder와 linear head를 validation checkpoint에서 고정한다. 그 linear head가 학습 latent 위의 affine prior다. 원래 raw feature에 대한 affine와는 다르므로 **latent-affine PP 확장 후보**로 명명한다.
- causal 16-recording GRU가 prior의 bounded residual을 학습한다. bound .25/1/3을 seed42 validation으로 선택 후 42–46 모두 학습한다. direct GRU에는 동일 encoder/history와 base prediction feature를 제공한다.
- early history는 당시 첫 관측 반복으로 padding한다. 실제 첫 기록 이전의 상태를 관측했다고 주장하지 않는다.
- 각 seed의 encoder도 독립 학습했으며 head만 여러 seed 돌린 결과가 아니다. 다만 PP와 direct의 encoder 공유는 representation 통제용이다.

## v8의 변경과 해석 제한

처음 최대 10개 관측 중 시점 t까지 이용 가능한 것만으로 log RMS baseline을 만들고, 절대 baseline 및 현재 대비 변화량을 입력했다. 초반 t<10에도 미래 기록은 보지 않는다.

validation Bearing3_2의 50/60/70/80/90% 관측 위치를 오프라인 평가 지점으로 정했다. 이 비율은 validation label 평가행 선택용이며 test 입력에 제공하지 않는다. test의 최종 수명 비율을 알고 예측한 것이 아니다. 모든 head/encoder의 validation selection에 같은 5점을 썼다.

v7은 validation 전 행, v8은 5점이므로 representation과 선택 프로토콜이 함께 변했다. 효과 분해 ablation 없이 둘 중 무엇이 개선 원인이라고 단정하지 않는다. PP가 실패한 상태라 이 후보를 제출용 개선 모듈로 채택하지 않았다.

## v9: 왜 크게 발산했나

학습 진행률은 q=(elapsed+50)/(elapsed+50+RUL), 모델 score f의 sigmoid가 q를 예측한다. 복원식은 (elapsed+50)*exp(-f)다. 50초는 기존 stride5 기록 간격에 대응하는 0시간 안정화 상수다. f는 공통 수치 안전 범위 [−6,6]으로 제한하며 train 최대 RUL cap은 쓰지 않는다.

v9의 진행률 target과 decoder는 수학적으로 일치한다. 과거의 inverse-softplus 초기화 불일치를 반복한 것은 아니다. 그럼에도 진행률 일반화 오차가 시간척도 복원 과정에서 크게 증폭됐다. 예측 f 오차 Δf는 RUL에 exp(−Δf)의 배율 오차를 만든다. 진행률이 정확하다는 보장 없이 이 decoder를 사용하면 안정적인 RUL 모델이 아니다.

validation 오차가 작았던 seed도 test에서 실패했다. v9 PP 개별 seed R²는 −70.219, −62.965, −77.228, −8.567, −104.066이다. 단순 ensemble 우연이나 한 seed 발산 문제가 아니다. **현 단계에서는 elapsed 비례 지수 decoder를 FEMTO 기본 head로 채택하지 않는다.**

## 이번 결과가 지지하는 판단

1. 원시 진동 CNN을 추가하는 것만으로 현재 PP의 일반화 실패가 해결되지 않는다.
2. 현재 validation 한 bearing에서 잘되는 정도가 test 성능을 안정적으로 예측하지 못한다. 이것은 validation label로 튜닝을 안 했다는 문제가 아니라 검증 unit의 대표성 문제를 포함한다.
3. v7/v8에서 많은 seed가 residual epoch0를 선택했다. 현재 학습/검증 데이터가 prior correction의 효과를 지지하지 않는 경우가 많다. residual bound만 키우는 다음 실험은 우선순위가 낮다.
4. 이 결과로 FEMTO의 양수 성능이 불가능하다고 증명한 것은 아니다. raw encoder가 조기 checkpoint를 선택한 경우가 많아 안정된 열화 표현을 학습했다는 근거도 부족하다.

다음 구조 연구를 한다면 waveform representation의 train-unit-held-out 전이부터 검증하고, label-free pretraining 또는 고장 형태별 표현을 분리한 뒤 head를 붙이는 것이 타당하다. 현재 5+1 한 validation에서 NN 크기/decoder를 반복 변경하는 방식은 멈춘다. 새 representation이 unit-held-out에서도 유효하다는 증거 없이 더 복잡한 PP라고 이름 붙이지 않는다.

### v10–v12가 추가로 배제한 가설

v10은 complete Learning bearing의 마지막 건강지표로 조건별 threshold를 만들고, causal robust slope로 도달시간을 계산했다. train-only pseudoendpoint grid로 선택했지만 test R²는 −12.485였다. 예를 들어 Bearing1_4는 임계값 근처여서 12초로 예측한 반면 정답은 339초로 방향은 맞았지만, Bearing1_5/1_6은 임계값에서 멀어 14520/9685초로 예측했고 실제는 1610/1460초였다. 하나의 vibration threshold가 모든 고장 모드를 나타내지 않는다.

v11은 threshold를 없애고 test 궤적을 같은 조건의 Learning 궤적 시점에 정렬했다. RMS·band·단/장기 변화, history, 이웃 수 및 time scale을 Learning leave-one-unit-out으로 선택했다. Bearing1_3, 1_4, 2_3, 3_3에는 비교적 맞았지만 1_5/1_6 및 condition2의 다수 bearing을 초기 열화로 오인했다. **관측 궤적 유사성만으로 future failure mode를 식별할 수 있다는 가설도 지지되지 않았다.**

v12는 Bearing3_2까지 학습에 포함한 all-six raw-CNN refit이다. epoch는 이전 v8 validation에서 얻은 중앙값 2로 고정했고 test로 고르지 않았다. 5 seed 모두 R² −2.01 이하였고 ensemble −2.332였다. 기존 v8의 양수 0.075는 validation bearing을 학습에 더 넣으면 자동으로 안정화되는 결과가 아니었다.

추가로 test label을 사용한 **진단용** leave-one-test-bearing-out Ridge를 v8 prediction, elapsed, condition에 적용했지만 R²는 각각 −.200, −.114, 조합 −.186, 전체 −.261이었다. 이것은 정식 모델 결과가 아니며 test label을 썼으므로 성능 표에 넣지 않는다. 다만 단순 선형 gate가 어느 bearing에서 prior/NN이 맞을지 구분할 가능성이 낮다는 진단이다.

현재 가장 정직한 FEMTO 처리 방법은 PP가 prior를 승인하지 않고 v8 direct ensemble로 빠지는 것이다. 이때 framework 결과는 0.075지만 개별 seed는 모두 음수이므로 강건한 해결로 선언하지 않는다. FEMTO를 PP의 성공 데이터셋으로 포함하려면 고장 모드 또는 미래 regime을 구별하는 추가 관측이 필요하거나, 6-event 소표본에 맞는 survival/latent-state 모델을 별도 연구 단계로 다뤄야 한다.

## 검증 및 저장 위치

- cache 표본 3개가 원본 acceleration 열 4/5와 byte-level float 값으로 일치하는지 통과.
- CNN 출력 dimension, PP zero-residual 초기화, nonzero bounded correction 검사 통과.
- 진행률 target/decoder round-trip 검사 통과(수치 clip 비활성 구간).
- 기존 v4에서 causal feature 미래 변경 불변성은 확인했지만 이번 CNN 전체 pipeline에 대해 exhaustive future-deletion test를 했다고 주장하지 않는다.
- `experiments/femto_waveform_pp_v7.py`: 기본 v7, `--reference` v8, `--progress` v9.
- `results/femto_waveform_pp_v7/`, `v8/`, `v9/` 이름의 각 폴더에 encoder checkpoint, encoder/head loss 곡선, test 전에 저장한 selection manifest, seed별 예측, results.json, unit별 오차/seed 통계 audit.json을 저장했다. 정확한 폴더명은 공통 접두어 `femto_waveform_pp_`를 포함한다.
- v10: `experiments/femto_health_threshold_pp_v10.py`, `results/femto_health_threshold_pp_v10/`.
- v11: `experiments/femto_trajectory_transport_pp_v11.py`, `results/femto_trajectory_transport_pp_v11/`. 최초의 큰 grid는 결과를 만들기 전에 중단했고, 저장된 manifest는 축소한 96개 train-only 후보 실행이다.
- v12: `experiments/femto_waveform_all_learning_refit_v12.py`, `results/femto_waveform_all_learning_refit_v12/`.
- NN 학습 설정은 validation으로 선택했지만 데이터셋은 이미 관측된 개발 test다. 봉인 확증이라고 주장하지 않는다.
