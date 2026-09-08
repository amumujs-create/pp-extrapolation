# PP 후속 모델 인계: 코드·결과 기반 문제 진단

범위: 최근 Na-ion/Zn-ion BQ → regime-scale → RBF-regime 실험. 전체 PP 저장소의 다른 데이터셋 결과까지 동일 문제라고 일반화하지 않는다. 이번 작업은 진단이며 모델·기존 결과는 변경하지 않았다.

## 우선순위 1 — 비교 모델 간 학습 정보와 출력 범위가 다름 (확정)

`experiments/znion_rbf_regime_pp_confirmation_v2.py:20,25`는 train과 validation을 합친 dev의 descriptor와 실제 log-EOL을 RBF memory로 저장한다. `:26`의 MLP는 train만 gradient 학습하며 validation은 checkpoint 선택에 사용한다. 장수명 validation 셀의 실제 수명을 PP만 prototype label로 직접 활용한다. Test label 직접 누수라고 단정하는 것은 부정확하지만, 두 모델의 supervision 사용 방식이 다르다.

`experiments/plain_mlp_ablation.py`는 MLP 출력을 [0, max(train RUL)]로 제한해 왔다. 실제 상한은 367 cycle이고 v3 장수명 `446-1`은 최대 정답이 740이다. 다만 사후 감사에서 MLP 예측이 상한에 닿은 seed는 0/5였고 clipping 제거 전후 점수가 같았다. 따라서 이것은 잠재적 설계 문제이지만 현재 v3 우세의 실제 원인으로 증명되지 않았다.

필요한 비교: 동일 development unit/EOL label/초기 prefix 정보를 양쪽에 제공하고 내부 validation으로 선택한 뒤 동일 재학습 절차 적용; MLP 양의 출력은 허용하되 학습 최대치 상한 제거; prefix descriptor 기반 EOL head도 대조군에 포함. 기존 MLP는 width32, tanh, lr5e-4, weight_decay2.0 한 설정이므로 최근 Zn-ion 결과가 충분히 튜닝된 NN 대비라는 주장도 미검증이다.

## 우선순위 2 — 실험은 매 시점 관측을 받는 online RUL (확정)

`znion_bq_confirmatory.py:70-85` 및 `naion_prefix_gate_eval.py:features_at`은 평가 시점 i의 실제 SOH와 i까지의 관측 history를 사용한다. 152-cycle에서 관측을 중단하고 모든 미래 RUL을 예측하는 open-loop forecast가 아니다. 현재시점 예측으로는 causal하며 이것 자체가 미래 누수는 아니다.

학습 행의 age<=152, 평가 행의 age>152이므로 age 좌표상의 support 외삽이다. health 범위까지 완전히 분리되었다거나 RUL target이 모두 학습 범위 밖이라는 뜻은 아니다. 전체 개발 파이프라인의 validation tail 사용도 명시해야 한다.

## 우선순위 3 — 독립적인 장수명 regime 정보가 부족 (관측 증거 있음)

RBF memory의 장수명 `442-2` EOL은 1010이고 일부 평가 셀의 life_hat도 거의 정확히 1010이다. 이 조건에서 RBF attention은 사실상 한 prototype의 수명을 전달한다. 장수명 셀을 통째로 memory와 affine fitting에서 제외한 최근 LOO 실험에서 해당 R²=-2.340이었다.

단, 그 LOO는 affine-only BQ이고 배포 재생은 학습 residual을 사용하므로 최종 모델의 완전한 nested CV 점수로 인용하지 말 것. 단일 prototype 의존성의 경고이지 최종 모델 전체가 같은 점수를 낸다는 증거는 아니다.

## 우선순위 4 — 통합 구조에 대해 과장된 설명 (확정)

- BQ 단독은 margin=0에서 RUL=0을 만족하지만 lifetime 경로와의 혼합은 보장하지 않는다. alpha1000 v3 `446-1` EOL 예측이 약79 cycle이다.
- gate는 `sigmoid(prefix_slope/1e-4)`의 고정 규칙이다. 학습된 regime-transition hazard나 동적인 regime detector가 아니다. 평가 tail 동안 gate는 고정되어 있다.
- RBF는 정규화 거리 기반 memory regression이다. 현재 attention distance와 gate를 NN과 end-to-end 공동 학습하지 않는다. 이것만으로 transformer 수준의 학습 attention 또는 독창성을 주장할 수 없다.
- RBF head는 결정론적이나 BQ residual은 NN이다. seed42 단일 실행과 재학습 분산 제거는 다르다.

최근 boundary attenuation 후보 margin/(margin+tau)는 development LOO에서 tau=0이 선택돼 기각됐다. 경계 보장과 정확도를 동시에 해결했다는 주장은 불가하다.

## 우선순위 5 — data provenance / 독립성 감사 결과

v2/v3 실행 코드에 이전 cohort와 후보 내부 SHA-256 중복 검사를 추가했다. 별도 사후 감사에서는 29개 파일의 원본 SHA와 80%-SOH EOL까지의 정규화 궤적 fingerprint를 함께 비교했다. 이미 알려진 `438-3`/`439-1` 한 쌍만 원본과 처리 궤적 모두에서 중복이었고, 새 중복은 발견되지 않았다. v3 실행 결과에도 exact duplicate와 candidate-internal duplicate가 모두 없었다.

이것은 로컬 파일과 처리 궤적의 중복 감사다. 서로 다른 fingerprint가 독립적인 실험 provenance를 증명하지는 않는다. 기존 `438-3`/`439-1` 중복이 upstream 원본 자체의 문제인지 다운로드/저장 경로 오류인지는 여전히 확인이 필요하다. 이전 다운로드 로그에서 HF tree 파일 크기와 받은 파일 크기가 다른 경우도 있어 origin revision, LFS oid, response URL, 내부 ID 재검증이 필요하다.

`read_cell`은 요약 시트를 쓰고 원 cycle을 1..N으로 재인덱싱한다. 결측 cycle이 있을 경우 실제 cycle 차이를 없앨 수 있다. 공식 preprocessor의 abnormal-cycle cleaning과 완전히 같은지 검증되지 않았다. read_cell의 None은 cycle10 부재도 포함하므로 전부 right-censored라 부르는 것도 부정확하다.

## 우선순위 6 — 성능 증거의 범위와 통계

v3 frozen pooled R²=0.558은 당시 사전 수치 기준을 만족한 결과다. alpha1000의 0.822는 이후 개발 결과다. Validation으로 alpha를 고르는 코드라도 탐색 방향·후보를 test를 본 뒤 설계했으므로 연구 전체가 test-independent였다고 할 수 없다.

v3 866행 중 장수명 셀이 741행(약85.6%)이다. pooled와 unit-macro 및 per-unit 지표를 함께 제시해야 한다. alpha1000에서도 `412-1` R²=-0.405이다. 예측 bias 약+15.2, RMSE19.8; 초반 tail 과대예측이 끝으로 갈수록 감소한다. 상수 offset만의 문제라고 볼 수 없다.

v2+v3 n=4에 대한 4/4 승리, RMSE 감소56.7%, bootstrap CI는 최근 개선 모델의 개발상 기술통계다. 모델/데이터/프로토콜 선택과 비독립성 때문에 확증적 유의성으로 해석하지 않는다. '한 셀 더 이길 때까지 추가해 p=.03125 확보'는 잘못된 방향이다. 표본 수·종료규칙을 먼저 정해야 한다.

## 후속 작업 순서

1. upstream origin revision/LFS oid와 preprocessing 동등성을 추가 점검.
2. online RUL과 prefix-only forecast를 별도 명세하고 같은 정보로 MLP/BQ/RBF 비교.
3. MLP 학습최대치 clipping을 제거한 공정 대조군 확보.
4. memory-only, BQ-only, 고정gate 혼합의 같은 split ablation으로 효과 원천 분리.
5. 그 뒤에만 lifetime quotient/학습gate 구조 개선과 여러 seed·unit holdout 검증.

현재 가장 확실한 결론은 'PP가 모든 NN보다 강하다'가 아니라 'BQ의 shrinkage와 장수명 memory 경로가 현 설정에서 유용하나, 정보 비대칭과 출력 상한 때문에 우세 크기를 아직 구조 효과로 분리하지 못했다'이다.
