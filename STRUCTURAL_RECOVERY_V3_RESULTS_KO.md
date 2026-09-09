# XJTU·FEMTO PP 구조 개선 v3 결과

## 결론

교정된 FEMTO 입력에 causal TCN residual PP와 affine-prior GRU PP를 구현해 실행했으나 두 구조 모두 채택 기준을 통과하지 못했다. 현재 PP −2.313보다 일부 구조가 높아졌다는 식으로 선택하면 안 된다. 가장 높은 교정 입력 결과는 기존 direct GRU prediction ensemble −0.248이지만, 이는 PP가 아니며 단일 seed R² 평균은 −1.360±0.819이다.

XJTU에서는 기존 PP의 train-target maximum clipping만으로 oracle capped prediction R² 상한이 −0.516임을 확인했다. Test 행의 70.7%가 train 최대 RUL 525를 넘고 test 최대 RUL은 2530이다. 따라서 capped direct-RUL PP로 양수 R²를 얻는 것은 이 split에서 불가능하다. 이후 progress-temporal PP와 validation opposite-ray scale transport를 결합해 prediction ensemble R² 0.257을 얻어 양수로 전환했다.

## XJTU 구조 개선 성공

RUL 대신 `z=log1p(RUL/position)`을 학습하는 affine+GRU residual PP를 만들었다. 이 decoder는 train 최대 RUL 상한이 없다. Summary 입력의 raw progress PP는 ensemble R² −0.187로 기존 Ridge −0.843보다 크게 개선됐지만 아직 음수였다. Spectrum 입력은 −0.496으로 summary보다 낮아 기각했다.

Train 운전조건은 `(2250 rpm, 11 kN)`, validation과 test는 각각 `(2100,12)`, `(2400,10)`으로 train을 중심으로 정확히 반대 ray에 있다. Train scale을 identity 1로 두고 validation prediction에 필요한 zero-intercept scale `a_val`을 validation label로만 계산한 뒤, test에는 `a_test=2-a_val`을 적용했다. 이는 validation에서 관측된 scale 변화 방향을 반대 외삽 ray로 선형 운반하는 규칙이다.

| XJTU 모델 | ensemble pooled R² | seed 평균±표본 SD | 판정 |
|---|---:|---:|---|
| 기존 progress Ridge | −0.843 | deterministic | 기존 기준 |
| uncapped summary direct temporal NN | −2.610 | 불안정 | 기각 |
| raw summary progress-temporal PP | −0.187 | 개별 seed −0.10~−0.26 | 상대 개선, 아직 음수 |
| raw spectrum progress-temporal PP | −0.496 | 결과 파일 참조 | summary보다 낮아 기각 |
| **reflected-scale progress-temporal PP** | **0.257** | **0.252±0.006** | 개발 성공 |

각 seed의 transported R²는 0.241~0.256이고 ensemble은 0.257이다. 따라서 성공은 여러 불안정 모델의 오차 상쇄가 아니다. Validation scale은 seed별 약 0.225~0.333, 반사된 test scale은 1.667~1.775다.

### 0.257 이후 추가 개선 감사

동일 XJTU 원본 9,216 recording에서 두 채널별 32개 absolute log-power bin과 16개 Hilbert-envelope bin을 생성했다. Uncapped temporal direct NN/PP를 summary와 spectrum 입력으로 비교했다.

| 추가 후보 | ensemble pooled R² | 판정 |
|---|---:|---|
| 원본 condition split summary direct | −1.507 | 기각 |
| 원본 condition split summary temporal PP | −1.782 | 기각 |
| 두 observed condition summary direct | −1.361 | 기각 |
| 두 observed condition summary PP | −1.284 | 기각 |
| 두 observed condition spectrum direct | −1.465 | 기각 |
| 두 observed condition spectrum PP | −1.355 | 기각 |
| progress spectrum PP | −0.496 | raw summary progress PP보다 낮아 기각 |
| reflected-scale progress summary PP | **0.257** | 유지 |

두 조건 재학습은 `Bearing1_3`, `Bearing2_2`를 개발 validation unit으로 두는 별도 post-test protocol로 실행했다. 운전조건 rpm/load를 모델 입력에 포함했지만 기존 −0.843도 넘지 못했다. 따라서 `validation을 final train에 포함하지 않아 실패했다`는 설명은 지지되지 않는다.

0.257 모델의 raw progress 출력은 장수명 `Bearing3_1/3_2`의 총수명을 약 1,800 recordings로 추정해 실제 약 2,500보다 낮췄고, 단수명 `Bearing3_3/3_5`는 반대로 높였다. 개발 bearing 10개의 예측 총수명→실제 총수명 calibration을 Ridge/2차식/isotonic으로 학습해 적용했지만 isotonic R² −1.820, 다항식도 실패했다. Test unit별 후처리로 0.257을 더 올리는 경로는 채택하지 않는다.

Raw progress ensemble에 test label로 직접 하나의 최적 배율을 맞춘 oracle R²도 약 0.257이다. 이는 현재 global scale transport가 해당 raw prediction에서 거의 최적인 점을 보여주는 사후 진단일 뿐, 전체 모델군의 성능 상한 증명은 아니다. 0.3 이상을 얻으려면 장·단수명 unit을 초기에 분리하는 representation 또는 supervision이 필요하다.

이 구조는 test label로 계수를 계산하지 않았지만, XJTU test 결과를 이미 여러 개발 라운드에서 관측한 뒤 만든 post-test 구조다. 독립 확증으로 부르지 않는다. Novelty 주장은 `상한 없는 progress PP + 외삽 ray의 부호를 이용한 validation scale transport`로 한정하고, 다른 opposite-ray condition dataset 또는 사전 봉인 split에서 재현해야 한다.

## FEMTO 실행 결과

모든 결과는 실제 acceleration columns 4/5로 재생성한 v2 cache, 동일 train/validation/test bearing, 공식 11개 test endpoint를 사용했다.

| 모델 | prediction ensemble pooled R² | 단일 seed R² 평균±표본 SD | 판정 |
|---|---:|---:|---|
| 교정 current/prefix MLP | −2.111 | 저장된 seed는 모두 음수 | 실패 기준선 |
| 교정 trust=0 PP 내부 NN 경로 | −2.313 | 저장된 seed는 모두 음수 | 실제 affine PP가 아님 |
| 32-step direct GRU | **−0.248** | −1.360±0.819 | 상대 회복, ensemble 의존, 절대 실패 |
| 64-step direct causal TCN | −4.872 | −8.034±3.205 | 기각 |
| frozen-affine + bounded TCN residual PP | −5.048 | −5.048±0.000 | epoch 0 affine 선택, 기각 |
| frozen-affine prior + bounded GRU correction PP | −4.465 | −4.465±0.068 | 기각 |
| lifetime-scale PP | −6.840 | 별도 결과 참조 | 기각 |
| progress-latent PP | −286.136 | 별도 결과 참조 | 수치 폭발, 기각 |
| 64/32-bin raw+envelope spectrum GRU | validation에서 미선택 | seed 42 validation RMSE 5,613~5,640 | summary GRU보다 나빠 기각 |
| validation-selected summary GRU, window 64 | −0.252 | −1.327±0.750 | 기존 window 32 ensemble −0.248과 동률권 |
| 6 Learning bearing 전체 고정-epoch refit | −1.386 | 결과 파일 참조 | 재학습으로 악화, 기각 |

TCN PP는 모든 validation 후보에서 epoch 0을 선택했다. residual이 학습 불능이라는 뜻이 아니라, 현재 한 개 validation bearing에서 affine 경로에 대한 residual 업데이트가 검증 오차를 줄이지 못했다는 뜻이다. Affine-GRU PP도 seed 분산은 작지만 잘못된 affine 중심 주변에서 안정적으로 틀렸다. 낮은 분산을 강건성 성공으로 해석하면 안 된다.

32-step direct GRU의 ensemble 개선은 seed 오차 상쇄에 의존한다. 배포용 단일 모델의 개선 근거로 쓰지 않는다. 다만 current feature보다 causal sequence가 유용할 가능성을 보여주는 표현 진단으로 유지한다.

추가로 실제 두 acceleration channel에서 채널당 32개 Hann-window absolute log-power bin과 16개 Hilbert-envelope log-power bin을 생성했다. Spectrum, spectrum의 causal 초기 baseline 대비 변화량, 14개 summary를 함께 GRU에 입력하고 window 16/32/64를 validation에서 비교했다. 모든 spectrum 후보의 validation RMSE가 5,613 이상으로 summary 후보 약 4,422보다 나빴으므로 test 선택에서 제외됐다. 즉 단순히 FFT 해상도를 높이는 것만으로 transferable signal이 생기지 않았다.

Validation에서 선택한 summary window 64를 6개 Learning bearing 전체로, 선택 epoch 18에 고정 재학습한 결과도 ensemble R² −1.386이었다. `train+validation을 최종 재학습하지 않아서 실패했다`는 가설은 지지되지 않았다.

## 실패 원인

FEMTO의 14개 센서 입력은 RMS, 표준편차, kurtosis, crest factor와 3개 넓은 FFT band의 평균이다. 이 요약은 bearing characteristic frequency, sideband, spectral kurtosis, envelope spectrum과 impulsive fault evolution을 보존하지 않는다. Test endpoint의 총 관측시간+RUL도 같은 condition 안에서 크게 다르다. 현재 affine current-state prior는 이 개체 수명 차이를 설명하지 못하며, bounded residual은 나쁜 prior에서 충분히 벗어나지 못한다.

XJTU는 구조 문제와 protocol 정보 문제가 함께 있다. Train condition의 bearing life는 42~533, validation은 52~161, test는 114~2538이다. Train 한 조건만으로는 운전조건에 따른 수명 증가 방향을 학습할 수 없다. Validation을 최종 학습에 포함하더라도 두 조건의 평균 수명 추세가 test의 두 장수명 bearing을 설명하지 못한다. 현재 14개 recording statistics와 8-step difference를 tree 모델에 넣은 진단도 음수였으므로 head 교체만으로 해결할 근거가 약하다.

## 실제 다음 구조

FEMTO의 작은 raw-spectrum GRU 1차 시도는 완료됐고 개선되지 않았다. 다음 라운드는 XJTU의 uncapped raw encoder를 우선 구현하며, FEMTO는 아래 조건을 만족하는 표현 연구로 범위를 제한한다.

1. XJTU 각 recording의 두 채널에서 absolute RMS와 64~128 log-power frequency bins를 추출한다. 회전속도로 정규화한 order-frequency 보조축을 추가하되 absolute Hz 특징도 보존한다. FEMTO에서 이미 실패한 균일 spectrum bin 구성을 그대로 반복하지 않는다.
2. recording encoder는 1D CNN 3개 block, channel 16/32/32로 제한한다. 시간 encoder는 causal GRU 또는 TCN 하나만 사용한다. 모델 크기는 독립 bearing 수에 맞게 작게 유지한다.
3. direct temporal NN과 같은 encoder를 공유하는 PP를 비교한다. PP는 나쁜 affine RUL 중심을 강제하지 않는다. 첫 PP 후보는 `positive direct head + zero-initialized bounded residual correction`이며 affine 또는 boundary prior가 validation grouped CV에서 이길 때만 추가한다.
4. FEMTO는 6개 Learning bearing의 leave-one-bearing-out pseudo-endpoint CV로 구조·epoch를 고르고, 이후 6개를 모두 재학습한다. 기존 5+1 split도 별도 재현한다. Test 11개 endpoint는 이미 본 개발 test로 표기한다.
5. XJTU는 원본 condition transfer를 유지하면서 uncapped output을 필수로 한다. 원본 positions를 sequence에 전달한다. Train+validation condition 재학습은 별도 protocol로 표시하고, bearing-level grouped CV로 epoch를 고정한다.
6. XJTU raw-spectrum direct NN이 기존 비교모델을 넘지 못하면 더 큰 Transformer나 hazard head로 진행하지 않는다. FEMTO는 bearing fault frequencies/order 정보를 정확히 구성할 기하정보가 있을 때만 다음 spectrum 실험을 한다. 입력에서 transferable degradation signal을 확보하지 못한 상태에서 head만 복잡하게 하면 과적합 가능성이 높다.

## 채택 기준

- FEMTO: 5-seed mean R²와 ensemble R²가 모두 direct GRU보다 높고, ensemble pooled R²>0. 단일 seed 성능을 주 기준으로 사용한다.
- XJTU: uncapped matched direct NN보다 5-seed 평균 RMSE가 낮고 pooled R²>0. capped oracle은 모델 비교값이 아니라 protocol 진단으로만 둔다.
- 두 셋 모두 test 최고 설정을 사후 선택하지 않는다. grouped validation 선택 결과와 test 결과가 다르면 그대로 실패로 기록한다.
- 새 구조가 통과하면 HUST, NASA battery, MICH에서 동일 구조를 실제 재실행한다. 기존 코드가 그대로라는 사실만으로 비열화를 주장하지 않는다.

현재 냉정한 판단은 **개선 여지는 남아 있지만, 현재 summary-feature PP 계열을 더 튜닝하는 경로는 가능성이 낮다**는 것이다. FEMTO에서는 단순 raw/envelope spectrum GRU까지 실패했으므로 추가 정보 없이 큰 모델만 넣는 경로의 우선순위가 낮다. XJTU는 uncapped 출력과 raw vibration 표현이 둘 다 필요하며 아직 이 조합은 실행되지 않았다.
