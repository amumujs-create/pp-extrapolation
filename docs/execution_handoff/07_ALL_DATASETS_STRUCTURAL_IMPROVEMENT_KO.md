# 전체 PP 구조 개선 실행 명세

작성: 2026-09-09. 상태: **코드 검토에 근거한 설계 및 실행 인수인계. 신규 모델 성능 결과가 아니다.**

사용자가 XJTU/FEMTO 외 나머지도 개선 대상으로 확대했다. 이 문서는 06번의 베어링 설계를 기존 12개 개발 설정으로 확장한다. 보류한 cohort는 열지 않는다. 새 고성능 모델이 이미 완성됐다고 보고하지 않는다.

## 1. 먼저 해결할 문제: 최종 PP가 하나의 실행 경로가 아니다

`experiments/final_modular_pp_evidence.py`와 아래 실제 runner를 확인했다. 기본 `fit_pp` 하나를 모든 데이터에 돌려 최종 PP라고 부르면 안 된다. 현재 개발 표는 데이터별 내부 모듈과 calibration이 다른 결과를 모은 것이다. 공통 구조가 모든 데이터에서 검증됐다는 증거와 구별한다.

| 설정 | 보존할 개발 표 R² | 실제 기준 runner | 우선 개선 가설 |
|---|---:|---|---|
| HUST | .958 | `experiments/hust_regime_transport_pp.py` | 외부 output transport의 이득을 내부 조건부 residual로 학습 |
| Virkler | .888 | `experiments/support_gated_cross_domain.py` | 균열 성장 가속 정보를 residual에 제공; 거리 감쇠의 과잉 여부 확인 |
| NASA battery | .584 | `experiments/nasa_causal_multiscale_pp.py` | 불규칙 시간간격과 짧은 history 신뢰도 처리 |
| Sunwoda | .934 | `experiments/bq_dual_scale_final_replay.py` | bounded quotient의 포화 진단 후 history 적응형 용량 |
| RWTH | .842 | 위와 동일 | 셀별 속도 차이와 작은 margin 구간의 민감도 분리 |
| MICH | .751 | 위와 동일 | 큰 residual이 필요한 구간에만 용량 부여; 무조건 감쇠 금지 |
| MATR2019 | .466 | `experiments/matr_pp_validation_calibration.py` | calibration 의존성과 비선형 scale 오차 분리 |
| MATR batch 2 | .862 | `experiments/matr_batch2_pp_five_seed_replay.py` | 조건부 scale과 health bias를 구별하는 내부 head |
| N-CMAPSS | .937 | `experiments/ncmapss_pp_multiscale.py` | 운전조건 변화와 열화 변화 분리; 성능 유지가 우선 |
| XJTU | .257 | `experiments/xjtu_reflected_scale_pp_v3.py` | 반사 scale 의존 제거, decoder 정합성, 단수명 bearing 붕괴 완화 |
| FEMTO | −2.313 | `experiments/femto_corrected_benchmark_v2.py` | 실제 PP 대조 복구, causal GRU와 grouped endpoint 선택 |
| NASA milling | .341 | `experiments/milling_boundary_quotient_route.py` | 희소 관측 slope 불안정과 경험적 경계 offset 분리 |

위 숫자는 기존 표의 참조값이지 이번 재실행 결과가 아니다. XJTU의 .257은 post-test 반사 보정, FEMTO −2.313은 trust=0, milling .341은 NN off다. 06번의 해석을 유지한다. Na/Zn-ion 등 추가 개발 데이터는 별도 exposure manifest에서 이미 개발에 사용했는지 확인한 뒤 보조 패널로만 추가한다. HNEI 등을 이름만 보고 자동으로 train에 넣지 않는다.

## 2. 먼저 구현할 공통 구조: 관측 이력에 따라 보정 용량을 바꾸는 PP

새로운 대형 Transformer 전체 교체보다 기존 PP의 성공 원리를 유지하면서 실패 원인에 해당하는 모듈만 바꾼다. 이것은 개선 가설이며 novelty 확정 주장이 아니다.

### 2.1 공통 history encoder

입력: 현재 특징 x_t, 과거 관측 H_{≤t}, 실제 Δt, padding mask, 관측 개수, 관측 기간, 알려진 운전조건 c_t.

첫 구현은 GRU16/32와 현재 multiscale 요약을 나란히 비교한다. 길이가 짧을 때 미래값으로 padding하지 않는다. packed sequence 또는 정확한 mask를 사용한다. 표준화는 train에서만 fit한다. 전체 수명으로 나눈 age는 금지한다.

길이가 다른 history의 단순 평균·기울기는 정확도가 다르다. 각 horizon의 실제 유효 표본 수와 실제 기간을 입력에 함께 넣는다. 두 시점 미만의 기울기는 0과 missing indicator로 나타낸다. 수치 0을 안정 상태라고 오인하지 않게 한다.

### 2.2 관측 경계가 있는 데이터의 head

현재 BQ 형태를 유지한다:

    y_hat = s_train * m_t * softplus(a(x_t) + r_t)
    r_t = B(z_t) * tanh(v(z_t) / B(z_t))
    B(z_t) = B_min + (B_max-B_min) * sigmoid(g(z_t))

여기서 m_t는 공식 경계까지의 비음수 margin, z_t는 causal history representation이다. 기존 BQ가 이미 시간 정규화한 y를 쓰면 s_train을 중복 곱하지 않는다. 현행 dual-scale가 이미 같은 목적의 용량 확대를 제공하므로 **현행 dual-scale가 직접 대조군**이다. basic BQ만 이겨서는 새 개선으로 채택하지 않는다.

B_min/B_max는 예측 범위가 아니라 latent correction 범위다. margin을 0으로 만들면 y_hat=0이 유지된다. margin이 없는 베어링에 임의의 RMS failure threshold를 만들지 않는다.

고정 bound B에 대해 softplus가 1-Lipschitz이므로 |y_hat-y_affine| ≤ s_train*m_t*B다. adaptive B≤B_max라면 동일한 상한에 B_max를 쓸 수 있다. **이는 오차의 상한이나 실제 RUL 정확도 보장이 아니다.**

이 구조가 도움되는 조건: 잘못된 affine를 작은 correction bound로 고칠 수 없으면서 history에 그 오차를 설명할 정보가 있을 때. 도움되지 않는 조건: history에 없는 미래 regime 변화, 틀린 경계, 또는 source와 target의 관측-수명 관계가 식별 불가능할 때.

### 2.3 경계가 없는 데이터의 head

06번의 decoder D0–D3를 먼저 비교한다. 공통 기본 후보는 y_hat=s_train*softplus(a(x_t)+r_t). affine 초기화는 inverse-softplus(y/s_train)에 fit한다. 시간 t를 출력에 강제 곱하지 않는다. 현재 최고 경로는 대조군으로 보존한다.

Direct NN도 동일 encoder/decoder/입력/학습 예산을 받는다. affine를 사용한 효용과 history 표현을 추가한 효용을 분리한다. 'gate'라는 이름을 붙였다고 신뢰 확률이나 사전 승인기가 되는 것은 아니다.

### 2.4 residual magnitude와 confidence를 혼동하지 말기

history가 크고 안정적인 correction을 지지할 때 용량을 늘리는 B와, 정보 부족 때문에 correction을 줄이는 confidence는 다르다. 최초 구현은 B만 학습한다. uncertainty head는 nested train OOF target을 만들 수 있을 때 별도 ablation으로 추가한다. 거리 또는 train residual만으로 uncertainty 정답을 만들지 않는다.

## 3. 데이터별 구체적 실험

### NASA battery: 시간 축과 관측량부터

현재 `cell_rows`는 slope를 실제 cycle 간격으로 나누지만 y는 `n-1-i`다. 원본 cycle이 연속인지 먼저 검사한다. 다르면 그것은 남은 관측 개수이며 cycle RUL과 다르다. 공식 목표를 확인해 label을 바꿀 때는 기존 .584 및 모든 비교모델을 새로운 프로토콜로 재평가한다. 둘을 같은 표에 직접 비교하지 않는다.

후보: H0 현재 multiscale, H1 실제 duration/count/missing 포함, H2 H1+GRU16, H3 H2+adaptive bound. 4-cell LOO를 보존한다. 선택한 한 fold의 변화가 전체 개선을 지배하는지 per-cell RMSE를 저장한다. 단 4개 cell이므로 대형 encoder는 우선하지 않는다.

### MICH / RWTH / Sunwoda: saturation이 원인인지 먼저 확인

같은 공동학습 데이터와 `full_part` refit 절차를 유지한다. 로그: raw residual, correction/B, margin, quotient, affine-only prediction, d y_hat/d raw, gate 값. margin quartile별 residual 포화 비율(|correction/B|>.95), RMSE, gradient 크기를 계산한다. 이것은 진단 기준이며 test에서 모델 선택하는 gate가 아니다.

후보: B0 현행 dual-scale, B1 학습 history bound, B2 B1+raw/log 혼합 loss, B3 B1+약한 시간 일관성 loss. 한 번에 모든 요소를 켜지 않는다. MICH 하나만 개선하고 Sunwoda/RWTH가 손상되는 경우 공용 개선으로 선언하지 않는다. dataset ID로 test 승자를 고르는 router를 추가하지 않는다.

### HUST / MATR batch 2 / MATR2019: transport를 내부로 넣을 때의 공정 비교

현행 HUST와 MATRb2는 validation에서 Ridge output transport를 학습한다. 새 모델에 calibration을 빼고 단순 raw 예측만 비교하면 실제 질문과 다르다.

별도 내부 후보: f_t=(1+κ*tanh(k(z_t,c_t)))*a(x_t)+B*tanh(r(z_t,c_t)/B), κ<1. 작은 scale 조절과 additive correction을 구별한다. k/r 마지막 layer를 0으로 초기화해 epoch0를 보존한다. κ∈{0,.25,.5}는 validation으로 고른다. 단, 이것도 scale-additive 비식별성이 남으므로 작은 penalty와 k=0 ablation을 포함한다.

최소 비교: 기존 raw PP / 기존 PP+기존 calibrator / 내부 후보 raw / 내부 후보+동일 calibrator / direct NN+동일 calibrator. 내부 후보 raw가 좋으면 구조 기여, calibrator에서만 이득이면 보정 기여로 보고한다. 사용한 validation label budget도 맞춘다.

MATR2019 코드에서 `final_ensemble`은 approval에 따라 raw 또는 calibrated인데 npz `prediction`에는 항상 calibrated가 저장된다. baseline snapshot 때 approval flag를 확인해 선택된 예측을 읽는다. 실제 현재 값이 틀렸다고 단정하지 말고 JSON과 예측 재계산의 일치를 검사한다.

### Virkler: 속도·가속도의 정보 가치

현재 입력/label 단위부터 확인한다. 실제 균열 관측으로부터 causal log-rate, 두 horizon slope 차이, 관측 기간을 추가한다. log-rate의 floor는 train으로만 정한다. near-zero rate의 비율도 기록한다. Paris 법칙의 상수나 지수를 test에서 추정하지 않는다.

비교: 현재 support-gated PP / 추가 history+같은 gate / 추가 history+감쇠 off / adaptive bound. support distance가 크다는 이유로 유용한 성장 가속 correction까지 지우는지 확인한다. 단조성은 균열 좌표에 대한 부분 제약과 시간에 따른 RUL 감소를 구별한다.

### N-CMAPSS: 무조건 복잡하게 만들지 말기

현재 .937과 경쟁모델 차이는 작다. 운전조건 c로 센서 s의 baseline μ(c)를 source train에서만 학습해 [s−μ(c), c]를 residual encoder에 주는 후보를 먼저 비교한다. raw sensor 경로를 제거하지 않는다. μ가 OOD 운전조건에서 잘못 외삽할 수 있으므로 raw/normalized 병렬 특징을 ablation한다.

TRA는 split 좌표이고 기존 모델 입력에도 들어간다. TRA를 몰래 제거하거나 test hard mask를 바꾸지 않는다. 같은 hard TRA rows로 비교한다. 큰 Transformer 및 공격적인 domain invariance loss는 나중이다. 운전조건 자체가 열화 속도에 정보를 주므로 무조건 제거하면 악화할 수 있다.

### NASA milling: 희소 관측의 속도 추정

현행 .341은 NN의 성공이 아니다. rate를 직전 두 점 차분 대신 causal 3/5점 robust slope로 추정하고 실제 관측 간격, slope SE 또는 bootstrap spread를 함께 준다. available points<3일 때 fallback을 명시한다. slope floor 후보는 train/validation으로만 정한다.

대조: 기존 deterministic route / robust slope deterministic / 같은 feature direct NN / robust slope+BQ residual. +.03 offset과 NN correction 효과를 따로 비교한다. validation 4행/test10행이므로 안정적인 NN 기여를 찾을 수 있다는 보장은 없다. material을 가렸다고 domain adaptation이 된다는 주장은 하지 않는다.

### XJTU / FEMTO

06번을 그대로 실행한다. XJTU 반사 보정은 공용 모듈에 넣지 않는다. FEMTO는 수정된 진동 columns4/5만 사용한다. 현재 −2.313의 trust=0와 GRU −.248를 정확히 표시하고 동일 encoder의 affine+residual 대조군을 새로 만든다.

## 4. 실행자가 구현할 파일과 계약

다음은 **아직 구현되지 않은 인터페이스**다. 이름을 완료된 실행 파일로 보고하지 않는다.

1. `experiments/pp_all_routes_manifest.py`: 위 12개 route의 code/config/cache hash, y 단위, train/val/test row_id, seed, postprocessor, refit 규칙, exposure status를 JSON으로 저장. 기준 npz와 JSON score 불일치 시 중단.
2. `src/pp_extrapolation/causal_history.py`: duration/count/mask-aware history와 encoder. 함수에 test y를 받지 않음.
3. `src/pp_extrapolation/adaptive_temporal_pp.py`: 공통 adaptive residual와 optional scale head. 기본 설정에서 기존 route를 대체하지 않음.
4. `experiments/pp_all_routes_v4.py`: registry로 loader/fit/predict/evaluate 분리. `--dataset`, `--stage audit|select|evaluate`, `--output` 제공.
5. `experiments/pp_all_routes_report.py`: baseline/candidate/direct NN 행 정렬 확인 후 delta pooled R², unit RMSE, seed statistics, paired cluster bootstrap 저장.

역할을 순서대로 맡길 때 첫 실행자는 manifest와 adapter 계약을 완성한다. 두 번째는 공통 encoder/head 및 단위 검증, 세 번째는 selection/five-seed runner를 구현한다. 동일 파일을 각자 다른 구조로 동시에 수정하지 않는다. 다음 실행자는 이전 단계 산출물과 미완료 목록부터 읽는다.

## 5. 학습과 채택 규칙

- 원래 성공 route의 freeze/refit/scale을 복제하는 baseline과 새 공통 프로토콜 결과를 별도 저장한다. 기존 BQ의 train+val refit을 00번의 no-refit 규칙 때문에 조용히 없애지 않는다. 프로토콜을 바꿀 때 모든 비교군에 동일 적용한다.
- 원본 최고 설정을 복구한 후 후보 선택은 train/validation만으로 한다. 후보 1개를 확정한 뒤 test를 평가한다. 이미 본 test이므로 개발 결과다.
- 후보 폭증을 막기 위해 decoder → representation → adaptive bound → loss → scale 순서. 단계마다 현행 baseline 포함, seed42 소규모 screening 후 상위 후보를 42–44로 검증. 마지막 후보는 42–46 모두 재학습.
- loss의 기본 raw-RUL MSE는 train-only s로 정규화. log loss λ∈{0,.1} 및 pair loss λ∈{0,.03}는 별도 단계. 시간 일관성은 실제 run-to-failure label에서만 적용하고 hard constraint로 쓰지 않는다.
- 학습 unit 균형 가중치와 validation pooled MSE는 목적이 다르다. 둘 다 기록하고 validation 주 선택 지표는 3-seed 평균 pooled MSE로 사전 고정한다. pooled R²는 데이터셋 안에서 계산하며 초/분/cycle을 섞은 전체 pooled R²는 만들지 않는다.
- 모든 후보에 epoch0 checkpoint 포함. cap-hit, positivity clamp-hit, saturation, train/validation loss, gradient norm, selected epoch 저장. validation이 개선 중인 max-epoch 후보만 같은 규칙으로 연장한다.
- 기존 PP, matched direct NN, 해당 설정 최고 비교군을 같은 입력 정보로 비교한다. GRU/history를 PP에만 추가해 구조 자체가 우수하다고 하지 않는다.
- 채택 여부는 validation에서 정한다. test는 regression 진단과 보고에 사용한다. test 결과로 어떤 dataset만 기존 모델로 되돌린 묶음을 만들면 retrospective portfolio라고 별도 표시한다.
- 전부 이긴다는 보장은 없다. 모든 12개에 대해 개선/유지/악화/미실행을 빠짐없이 보고한다. 상위 저널의 논거는 각 데이터 승률뿐 아니라 어떤 모듈이 어떤 조건에서 유효한지 재현 가능한 근거다.

## 6. 실험 우선순위와 중단 조건

P0: 12개 baseline manifest와 score 일치 검사. 새로운 후보 학습 이전.

P1: NASA battery + MICH/RWTH/Sunwoda + XJTU/FEMTO. 서로 다른 head/정보 문제를 대표한다. 여기서 실제 이득을 확인한 공통 모듈만 나머지에 확대한다.

P2: MATR2019/MATRb2/HUST의 내부 scale head; Virkler history; milling robust slope.

P3: N-CMAPSS 운전조건 분리와 전체 matched regression. 작은 이득 때문에 복잡도를 크게 올리지 않는다.

한 단계에서 세 seed validation 평균이 기존보다 나쁘고 loss/gradient/포화 로그에 구현 결함이 없으면 무작정 모델을 키우지 않는다. 후보를 실패로 보존하고 다음 원인 가설로 이동한다. 최종 5-seed test를 본 뒤 새로운 후보를 고르면 새 버전으로 표시한다.

## 7. 필수 검증과 결과 표

필수 검증: 미래 데이터 변경 시 과거 예측 불변, padding 길이 불변(nonzero residual/direct 둘 다), epoch0 affine 복원, margin0 exact zero, B≤B_max, 단위 변환 일관성, fold별 train-only normalization, row_id 정렬, PP/NN 동일 label과 평가행.

최종 표 열: dataset / baseline route hash / baseline pooled R² / candidate pooled R² / Δ / 5-seed R² mean±SD / matched NN / 개선 unit 수 / 악화 unit 수 / runtime / validation 선택 / 개발 또는 봉인 상태.

unit bootstrap은 unit을 resample한 뒤 각 draw의 pooled R² 차이를 다시 계산한다. seed를 독립 unit처럼 세지 않는다. FEMTO endpoint처럼 unit당 한 행인 경우와 연속 tail 수천 행인 경우를 명시한다. 표본이 적으면 CI가 넓을 수 있으며 그것도 결과다.

## 8. 논문에서 남길 수 있는 모델링 주장

검증할 주장은 '어떤 데이터도 이기는 NN'이 아니라 '관측 가능한 boundary와 causal history를 이용해 affine prior의 correction capacity를 적응시키며 외삽 중 bias–variance tradeoff를 개선한다'이다. 이 주장을 쓰려면 adaptive capacity 대 현행 dual-scale, same encoder direct NN, condition-scale off, loss off, no-history의 matched ablation이 필요하다.

수학적 contraction 상한, 개별 unit 손해, 입력 신뢰도와 포화의 관계를 함께 보인다. 위 구조들은 기존 gated residual/BQ와 겹치므로 새 이름만으로 novelty가 확보되지 않는다. 성능 검증 후 관련 선행연구 대조가 별도로 필요하다.
