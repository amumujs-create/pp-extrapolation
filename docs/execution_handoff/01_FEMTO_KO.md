# FEMTO 실행 명세 — 최우선

## 목표
올바른 진동 입력으로 공식 truncated endpoint RUL을 예측한다. 그 뒤 동일 입력의 PP와 MLP/monotone NN 성능을 비교한다. 양수 개선 가능성을 데이터 오류 수정 이전에 단정하지 않는다.

## 읽을 파일
- `../ca-css-ncmapss/femto_bearing_loader.py`: `_rows_from_bearing_dir`, `load_femto_phm2012`, `FEATURE_COLS`, `OFFICIAL_TEST_RUL_S`
- `experiments/femto_pp_prospective.py`: 기존 split
- `experiments/remaining_failures_structural_pp.py`: 미래 baseline 문제와 기존 prefix 모델
- `src/pp_extrapolation/model.py`: fit_pp/predict

## 확인된 오류
실제 Learning_set/Bearing1_1/acc_00001.csv는 2560×6.
첫 두 열 std=0, RMS=9,39. 기존 cache도 rms_h=9,rms_v=39, FFT=0.
마지막 두 열 RMS는 약 0.561746,0.435801.
기존 로더는 arr[:,0],arr[:,1]을 사용한다. 6열 PRONOSTIA schema에서는 앞 4개 시간 열과 뒤 2개 진동 채널을 구분해야 한다.

## F0. 원본 schema/경로 감사
신규 `experiments/femto_sensor_adapter_v2.py` 구현.
1. 접근 whitelist: `data/femto/raw/Learning_set`, `data/femto/raw/Test_set`만. `Full_Test_Set`는 glob/rglob/feature 계산에 사용하지 않는다.
2. Learning 6개와 official truncated Test 11개 bearing의 명시적 경로 목록을 만든다. 없는 이름은 오류로 중단한다. 다른 경로로 자동 fallback하지 않는다.
3. 각 bearing 첫/중간/마지막 파일로 열 수 확인. 6열이면 channels=(4,5), timestamp=(0,1,2,3). 다르면 파서가 추측하지 않고 schema audit 실패.
4. delimiter는 쉼표/세미콜론을 확인해서 기록한다. 파일번호로 숫자 정렬, 중복 파일번호 assert.
5. `schema_audit.json`: raw path, sampled file hash, shape, column std/RMS, selected channels 저장.

## F1. 새 캐시·시간/라벨
- versioned cache: `data/femto/features_sensor_v2.npz` + JSON manifest. pickle 캐시 재사용 금지.
- stride=5. subsample 인덱스에 원본 마지막 recording을 추가해 마지막 관측이 빠지지 않도록 한다.
- 실제 시간 간격을 원본 timestamp로 점검한다. 10초 가정을 확인 후 일관된 seconds 단위 사용.
- train/validation RUL=(실제 원본 마지막 recording time-current time).
- Test 공식 잔여수명은 **원본 truncated 마지막 관측 시점**에 연결한다. 중간 시점 RUL은 평가용으로만 `official_endpoint_RUL+end_time-current_time`.
- 마지막 stride 파일을 원본 종료 시점으로 오인하는 현재 로더의 label shift를 제거한다.
- 공식 answer mapping이 없는 bearing에 0을 대입하지 말고 중단한다.
- 수치가 예전 label hash와 달라져도 원인이 timestamp/endpoint 보정이면 새 protocol version과 delta report로 남긴다. 예전 점수와 직접 승패 비교 금지.
- split IDs 유지: train [11,12,21,22,31], validation [32], test [13,14,15,16,17,23,24,25,26,27,33].

## F2. causal feature
각 recording 양방향 RMS, std, kurtosis, crest factor, 기존 FFT 3 bands 생성. FFT band 정의와 scaling은 비교 모델 전부 동일.
prefix 특징: elapsed seconds, log1p(elapsed), current statistics, baseline-relative statistics, 최근 8/32개 관측 slope/variation, valid_history_length, mask.
- baseline(t)=median(first min(10,t+1) observations). 첫 10개 전체를 t<9에 쓰지 않는다.
- slope 분모는 관측 개수가 아닌 **실제 경과 초**. 0분모 slope=0과 valid flag.
- 짧은 history는 관측 가능한 부분만 쓴다. padding하면 mask 제공. 미래/전체 trajectory 정규화 금지.
- train으로 학습한 스케일러를 다른 split에 적용.

## F3. 첫 공정 비교
신규 `experiments/femto_corrected_benchmark_v2.py` 구현.
동일 rows/features의 네 모델:
A Ridge (alpha=[.1,1,10,100,1000,10000])
B plain MLP (공통 grid)
C 기존 affine+NN PP (공통 grid)
D B/C의 causal-history 버전. current-only vs history-only 정보 효과를 먼저 분리.
추가 monotone NN은 condition 열에 제약을 걸지 않는다. 아래 시간 일관성 loss를 같은 정보로 적용한 비교군을 사용하는 편이 낫다.
최종 endpoint pooled R², 11 bearing 각각의 AE, RMSE, seed mean±SD, ensemble R² 저장.
Ridge/MLP가 PP를 이겨도 모든 결과 보존. 새 NN의 개선을 PP만의 novelty로 부르지 않는다.

## F4. 실제 PP 개선 후보: prefix 시간 일관성
F3의 데이터/label 검증이 통과한 뒤만 시작.
train 한 unit의 두 prefix t_i<t_j를 sample. 물리적 고장시점이 같으므로
`RUL_i-RUL_j ≈ t_j-t_i`.
`L_cons=Huber((r_i-r_j)-(t_j-t_i)) / target_scale`에 맞춰 단위 정규화 후 loss를 구현한다. 정확히는 residual을 target_scale로 나눈 뒤 Huber를 적용한다.
`L=L_supervised+lambda_cons*L_cons`, lambda∈{0,.01,.1}. early crossing/censoring row를 섞지 않는다.
이 loss는 시간 일관성을 높일 뿐 총수명 offset을 알려주지는 않는다. test RUL로 pair를 만들지 않는다.
Ablation: PP λ=0/선택λ, MLP λ=0/선택λ, 같은 prefix/예산.

## F5. 추가 구조는 필요할 때만
corrected F3/F4 모두 부진하고 validation에서도 temporal 특징이 유효하면 masked TCN/GRU history encoder를 PP residual로 추가한다.
입력 범위는 관측된 prefix뿐. 다음 특징/증분 예측으로 train 내부 사전학습을 할 수 있다. test trajectory로 사전학습하지 않는다.
비교군에 같은 encoder의 direct RUL 모델을 반드시 둔다.

## 구현할 CLI
```
python experiments/femto_sensor_adapter_v2.py --audit-only
python experiments/femto_sensor_adapter_v2.py --build-cache --stride 5
python experiments/femto_corrected_benchmark_v2.py --stage select --output results/femto_corrected_v2
python experiments/femto_corrected_benchmark_v2.py --stage evaluate --manifest results/femto_corrected_v2/selection_manifest.json
```

## 통과 조건
- 학습 파일의 raw channels RMS와 cached RMS allclose(rtol=1e-6).
- timestamp 열 교란이 진동 특징에 영향을 주지 않음(시간 특징은 별도).
- future suffix 교란/삭제 테스트 통과.
- original last test recording 포함, endpoint label=official answer 정확히 일치.
- test row_id 11개, 모델별 순서 동일.
- 이 조건 중 하나라도 실패하면 모델 순위와 양수 성공을 발표하지 않는다.
