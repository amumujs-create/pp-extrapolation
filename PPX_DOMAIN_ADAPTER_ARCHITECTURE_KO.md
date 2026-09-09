# PP-X 공통 core와 도메인별 feature adapter

## 목적

범용성은 모든 데이터에 같은 feature를 입력하는 것으로 정의하지 않는다. 각 도메인의 관측 단위와 열화 메커니즘에 맞는 **causal feature adapter**를 허용하되, 이후의 prior contract, transferability 판단, bounded correction 및 평가 계약은 공통으로 유지한다.

## 공통 인터페이스

각 adapter는 시점 t마다 다음을 반환한다.

    current_state     현재까지 관측된 건강 상태 표현
    causal_history    해당 unit의 시점 <=t 정보만으로 만든 sequence
    degradation_rate  단기/장기 변화 및 유효 관측기간
    context           운전조건 또는 실험조건
    margin            알려진 고장경계까지 거리, 없으면 None
    support_features  source support 거리 계산용 표현
    reliability_meta  관측 수, 결측, slope 불확실성

adapter의 fit은 train만 사용한다. validation은 feature family와 hyperparameter 선택에만 사용한다. test cohort 통계, test-prefix self-supervision, TTA 및 test unit 간 정보 공유는 금지한다.

## 도메인별 adapter

| 도메인 | feature adapter | 허용 prior |
|---|---|---|
| 배터리 | capacity margin, causal multi-horizon slope/curvature, cycle interval, protocol | boundary quotient, dual-scale residual |
| 균열 | crack length margin, log growth rate, acceleration, load context | Paris-compatible affine/quotient tail |
| 터보팬 | condition-normalized sensors, TRA, causal moments | latent regime/affine tail |
| 베어링 | waveform energy, envelope/order spectrum, impulsiveness, causal HI change | learned HI boundary 또는 survival prior; source evidence 없으면 abstain |
| milling | wear margin, robust causal slope, inspection interval, material | inspection boundary quotient |

## 공통 PP-X core

    z_t = Encoder(causal_history, context, reliability_meta)
    y_prior = Prior(current_state, degradation_rate, margin, context)
    y_corrected = Decoder(y_prior, B(z_t) tanh(r(z_t)/B(z_t)))
    route = TransferabilityGate(source-only evidence)
    y_hat = route.prior_weight*y_corrected + (1-route.prior_weight)*y_direct

동일 adapter와 encoder를 direct 비교군에도 제공한다. feature engineering의 효과와 prior의 효과를 분리한다. `prior_weight=0`은 PP-X framework가 입력 도메인을 거절하지 않고 부적합한 prior만 거절한 것이다. 논문에서는 prior improvement와 framework coverage를 별도 지표로 쓴다.

## FEMTO adapter 실험

회전체 전용 order/envelope adapter를 실제 구현했다. RPM별 1/2/3/4/5/6/8/10/12/16차 order power, envelope-order power, broadband power, spectral entropy, RMS, kurtosis와 crest factor를 두 채널에서 추출했다. 이 feature를 같은 GRU direct와 frozen-affine residual PP에 넣었다.

결과는 direct −1.347, PP −2.450이었다. 기존 waveform executor .075보다 낮아 채택하지 않았다. feature가 더 도메인 특화됐다는 사실만으로 target failure mode가 식별되지는 않았다.

두 번째 adapter는 source RUL label로 0→1 health indicator를 학습하고 causal cumulative-max projection으로 단조성을 적용했다. 현재 HI와 causal slope에서 경계 도달시간을 계산했다. Learning bearing leave-one-out pseudoendpoint로 feature/regularization/window를 선택했다. 결과는 −1.085였다.

마지막으로 waveform executor와 constrained-HI prior의 weight를 held-out Bearing3_2의 5개 prefix endpoint에서 선택했다. validation은 weight .5를 선택했으나 test R² −.051로 waveform-only .075보다 낮았다. soft gate도 채택하지 않았다.

산출물:

- `experiments/femto_order_feature_pp_v15.py`
- `results/femto_order_feature_pp_v15/`
- `experiments/femto_constrained_hi_pp_v16.py`
- `results/femto_constrained_hi_pp_v16/`
- `experiments/femto_ppx_soft_hi_gate_v17.py`
- `results/femto_ppx_soft_hi_gate_v17/`

## 현재 채택 상태

공통 core + 도메인 adapter 설계는 채택한다. 그러나 adapter 후보의 성능은 데이터별 validation/test 결과로 별도 채택한다. 현재 FEMTO는 waveform neural safety route .075를 유지한다. XJTU .257, milling .341, MATR2019 .466, NASA battery .584도 이번 신규 후보가 넘지 못해 기존 route를 유지한다.

향후 범용 개선은 adapter를 source-only로 사전학습하는 방향이 가능하다. 예를 들어 베어링 source waveforms에서 augmentation-consistent representation을 학습한 뒤 RUL head를 fine-tune할 수 있다. 이 경우에도 validation/test unit을 representation fit에 넣지 않고, 같은 pretrained encoder를 PP와 direct에 제공해야 한다.
