# PP-X 저성능 경로 구조 개선 감사

2026-09-09. 목표는 기존 고성능 경로를 보존하면서 낮은 pooled R² 경로를 새로운 PP 구조로 개선하는 것이었다. 신규 후보가 기존값보다 낮으면 최종 registry에 채택하지 않았다. 모두 이미 관측된 데이터의 retrospective development다.

## 결과

| 데이터셋 | 기존 PP-X | 새 후보 | 새 pooled R² | 채택 |
|---|---:|---|---:|---|
| FEMTO | −2.313 교정 기본 PP | transferability gate → raw waveform neural safety | **.075** | 채택 |
| XJTU | **.257** reflected-scale temporal PP | decoder-aligned adaptive residual PP | −1.580 | 기각 |
| NASA battery | **.584** causal multiscale PP | validation-selected target perturbation PP | .581 | 기각 |
| NASA milling | **.341** inspection-calibrated quotient | robust Theil-slope boundary PP | −1.375 | 기각 |
| MATR2019 | **.466** calibrated latent PP | calibration strength 진단 | 최대 약 .466 | 구조 이득 없음 |
| FEMTO 추가 adapter | **.075** waveform safety | order/envelope PP −2.450; constrained-HI PP −1.085; soft-HI gate −.051 | 기존 유지 |

따라서 최종 PP-X registry는 FEMTO만 갱신하고 나머지는 기존 route를 유지한다. 새 구조를 도입했다는 이유만으로 낮은 결과를 최종 모델에 넣지 않는다.

## XJTU

새 후보는 기존 `position*exp(score)`의 시간 비례 bias를 제거했다. train-only scale `s`에 대해 `RUL=s*softplus(f)`를 사용하고, affine 초기화를 정확한 inverse-softplus target에 fit했다. GRU residual capacity는 history로 .15에서 선택 bound까지 조절하고 train 최대 RUL cap을 제거했다.

validation MSE는 direct 후보에서 크게 낮아졌지만 condition-3 test로 전이되지 않았다. direct ensemble R² −1.285, PP ensemble −1.580이었다. affine PP는 epoch0가 선택되는 경우가 많았다. decoder 불일치가 기존 XJTU 실패의 유일 원인은 아니다. 기존 .257도 retrospective reflected-scale 보정이라는 한계를 유지한다.

산출물: `experiments/xjtu_aligned_decoder_pp_v4.py`, `results/xjtu_aligned_decoder_pp_v4/`.

## NASA battery

기존 robustness 결과에서 20% train-target perturbation이 test ensemble .594를 보였지만 그것은 test에서 확인된 사후 현상이었다. 새 실험은 noise level 0/.05/.1/.2를 각 fold validation으로 선택한 뒤 5 seed를 재학습했다. 선택 level은 fold별 .2/.2/.1/.1이었고 최종 R²는 .581이었다.

따라서 .594를 최종 개선값으로 사용하지 않는다. validation 선택 규칙으로 재현되지 않는 test 우연이다. 기존 clean PP .584를 유지한다.

산출물: `experiments/nasa_regularized_pp_v3.py`, `results/nasa_regularized_pp_v3/`.

## NASA milling

누적 OLS rate 대신 최근 2/3/5/8/all observations의 OLS 또는 Theil–Sen slope와 boundary offset을 validation MAE로 선택했다. validation은 window8, Theil, offset −.05를 선택했지만 test R²는 −1.375였다. 희소 validation case 하나가 material-shift test의 rate estimator 선택을 대표하지 못했다.

기존 .341 route는 offset +.03과 test material에서 NN residual off인 deterministic 결과다. 새 robust slope로 교체하지 않는다.

산출물: `experiments/milling_robust_rate_pp_v2.py`, `results/milling_robust_rate_pp_v2/`.

## MATR2019

저장된 base와 calibrated predictions 사이의 correction strength를 확인했다. base .257에서 full calibrated .466으로 개선되며, 그 이상 correction은 거의 plateau 후 악화한다. test 진단에서 1.25배 correction은 소수점 네 자리에서만 차이가 있고 validation으로 확정된 새로운 구조 이득이 아니다. 기존 .466을 유지한다.

## 모델링 결론

이번 결과는 하나의 새로운 decoder, regularizer 또는 slope estimator가 모든 저성능 도메인을 동시에 올린다는 가설을 지지하지 않는다. PP-X의 공통성은 동일 수식이 아니라 다음 실행 계약에 있다.

1. domain contract로 admissible prior family를 정한다.
2. source-only transfer evidence가 부족하면 prior를 거절한다.
3. 승인된 prior에는 bounded neural correction을 적용한다.
4. 각 route의 validation 선택과 test 평가를 분리한다.
5. 새 후보가 실패하면 기존 frozen route를 보존한다.

현재 통합 개발 결과는 12개 설정 모두 양의 pooled R²지만 FEMTO .075, XJTU .257, milling .341, MATR2019 .466은 약한 경로다. FEMTO 개별 seed가 모두 음수인 문제도 남는다. 상위 저널용으로 이 숫자를 숨기기보다 coverage와 정확도, single-fit robustness를 분리해야 한다.

다음 성능 개선의 우선순위는 모델 폭 확장이 아니다. XJTU는 condition별 complete bearing coverage, milling은 validation material/case 수, FEMTO는 failure-mode 식별 정보가 병목이다. 같은 데이터에서 test를 반복해 구조를 선택하면 숫자는 만들 수 있어도 일반화 증거는 약해진다.
