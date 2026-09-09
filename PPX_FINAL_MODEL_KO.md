# PP-X: prior-transferability adaptive extrapolation framework

## 최종 개발 모델의 정의

논문 본문에 사용할 축약된 최종 구조와 구성요소 채택/제외 규칙은 `PPX_FINAL_PAPER_MODEL_KO.md`에 고정한다. 이 문서는 그보다 넓은 개발 portfolio와 historical route를 보존한다.

PP-X는 하나의 수식을 모든 데이터셋에 강제하는 모델이 아니다. 관측 가능한 domain contract에 따라 prior를 구성하고, source의 transfer evidence가 부족하면 그 prior를 사용하지 않는 하나의 학습·실행 프레임워크다.

## 연구 범위: inductive extrapolation으로 고정

PP-X의 공식 범위는 처음 보는 unit/domain에 대한 **inductive extrapolation**이다. Test cohort는 학습, 표현학습, 정규화 통계, hyperparameter 선택 또는 gate threshold 선택에 사용하지 않는다. Test 입력은 확정된 frozen model의 forward prediction에만 사용한다.

공식 결과에서는 TTA, test-prefix self-supervised update, test-batch normalization/calibration, test unit 사이의 정보 공유, test prediction을 본 route 변경을 모두 제외한다. 각 test unit에 예측 시점까지 주어진 자기 자신의 causal history를 입력하는 것은 허용한다. 이때도 파라미터와 전처리 통계는 바뀌지 않는다.

    domain contract -> candidate prior -> transferability gate
                                      -> bounded neural correction
                                      -> neural safety executor when rejected

prior 후보는 known-boundary quotient, affine tail, causal multiscale latent state, regime-conditioned lifetime이다. gate의 출력은 확률로 해석하지 않는다. boundary가 알려졌거나 source OOF에서 prior가 matched direct model보다 안정적으로 좋을 때 prior 경로를 승인한다. complete group 수, regime당 group 수, OOF regret 또는 mode stability가 부족하면 neural safety 경로로 간다.

예측식의 일반형은 다음과 같다.

    y_prior = P(x, H, domain contract)
    y_PP = D(y_prior + B(H, support) tanh(r_NN(H)/B))
    y_PPX = g_transfer y_PP + (1-g_transfer) y_NN

현재 gate는 hard decision이다. FEMTO에서 `g_transfer=0`, 기존 승인 PP 경로에서는 1이다. 향후 soft gate를 사용하려면 source OOF에서 calibration해야 하며 test 성능으로 weight를 고르면 안 된다.

## 현재 통합 개발 결과

| 데이터셋 | PP-X 내부 경로 | pooled R² |
|---|---|---:|
| HUST | regime transport PP | .958 |
| Virkler | support-gated PP | .888 |
| NASA battery | causal multiscale latent PP | .584 |
| Sunwoda | validation-approved bounded boundary PP | .939 |
| RWTH | validation-approved bounded boundary PP | .878 |
| MATR2019 | validation-calibrated latent PP | .466 |
| MATR batch 2 | regime transport PP | .862 |
| N-CMAPSS | causal multiscale latent PP | .937 |
| MICH | support-adaptive dual-scale boundary PP | .751 |
| XJTU | reflected-scale temporal PP | .257 |
| FEMTO | transferability gate -> waveform neural safety | **.075** |
| NASA milling | inspection-calibrated boundary quotient | .341 |

현재 개발 registry에서는 12개 설정 모두 pooled R²가 양수다. Sunwoda·RWTH에는 fixed bound, MICH에만 support-adaptive dual scale을 쓰는 것이 최종 selection rule이다. 이것은 하나의 동일 신경망이 12개 데이터셋을 학습했다는 뜻이 아니다. PP-X의 domain contract와 route가 데이터별로 다르며 실험 protocol도 이질적이다. 기존 frozen prediction artifact를 묶은 개발 portfolio 결과다.

## FEMTO를 PP-X라고 부를 수 있는 이유와 제한

FEMTO에서도 모델이 PP-X 실행 흐름을 따른다. 먼저 known boundary, complete source coverage, regime coverage, source OOF transfer evidence를 검사한다. FEMTO는 명시적 failure boundary가 없고 condition별 complete bearing 수가 2개뿐이라 사전에 정한 최소 3개를 만족하지 못한다. 따라서 prior weight를 0으로 하고 waveform neural executor로 전환한다. test y는 route 결정 함수의 입력이 아니다.

이것은 “FEMTO에서 물리 prior가 NN을 개선했다”는 결과가 아니다. **PP-X가 부적합한 prior로 인한 음수 성능을 막고 예측 가능한 fallback을 제공했다**는 결과다. 최종 ensemble R² .075는 기존 corrected GRU ensemble −.248보다 높지만, 5개 개별 seed는 모두 음수다. 재학습 강건성이 해결됐다고 쓰면 안 된다.

## 논문 주장 범위

가능한 핵심 주장은 `domain-contract-conditioned prior portfolio with transferability-aware abstention`이다. 기존 PP의 boundary/affine residual 구조를 유지하면서 prior applicability 자체를 모델 실행 계약에 포함한다. 성능 주장은 development 12/12 positive coverage와 각 route의 matched 비교를 분리한다.

상위 저널 주장에는 다음이 추가로 필요하다.

1. transferability gate의 threshold를 현재 개발 데이터에서 고정하고 새로운 dataset/cohort에서 한 번 평가
2. always-prior, always-direct, validation-best, PP-X gate의 matched ablation
3. prior 승인 coverage, selective regret, pooled R², unit RMSE와 seed stability 동시 보고
4. XJTU reflected scale, milling offset, FEMTO safety route가 retrospective development라는 표시 유지
5. gate가 틀린 승인과 틀린 거절을 모두 보고하고 soft gate를 test에서 조정하지 않기

따라서 PP-X는 모델 확장과 FEMTO coverage까지 구현된 **최종 개발 버전**이다. 독립 외부 확증이 끝난 최종 논문 증거라는 의미는 아니다.

## 구현과 재현

- prior gate: `src/pp_extrapolation/transferability_gate.py`
- residual executor selection: `src/pp_extrapolation/executor_policy.py`
- 논문용 축약 구조: `PPX_FINAL_PAPER_MODEL_KO.md`
- FEMTO route: `experiments/femto_ppx_final_v14.py`
- 통합 registry: `experiments/ppx_unified_final_registry.py`
- FEMTO 산출물: `results/femto_ppx_final_v14/`
- 통합 산출물: `results/ppx_unified_final_v1/registry.json`
- 저성능 경로의 신규 구조 채택·기각 감사: `PPX_LOW_SCORE_STRUCTURAL_IMPROVEMENT_AUDIT_KO.md`
- 공통 core와 도메인 feature 계약: `PPX_DOMAIN_ADAPTER_ARCHITECTURE_KO.md`
