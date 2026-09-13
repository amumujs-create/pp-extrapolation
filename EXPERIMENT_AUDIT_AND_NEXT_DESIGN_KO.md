# 기존 실험 감사와 다음 설계 판단

새 학습 없이 현재 코드와 완료된 실험의 predictions.npz를 분석했다. EDIST는 사용자 요청으로 중단됐으며 Misata와 전체 집계는 미완료다. 기존 결과는 수정하지 않았다.

## 핵심 판단

현재 변형들의 성능을 제안한 메커니즘의 효과로 해석할 수 없다. 구현과 최종 재학습 절차부터 정리해야 한다. 신규 구조의 외부 우월성과 문헌상 독창성은 입증되지 않았다.

## 구현에서 확인한 문제

1. SRCIST 보정 우회: SourceRiskCalibratedModel은 mean/predict를 수정하지만 실제 predict_innovation_slope_transport는 model.samples를 호출한다. __getattr__을 통해 원본 samples로 전달되므로 보정이 최종 예측에 적용되지 않는다. 반환 validation_mse도 원본 값이다. 보정 효과라는 이전 설명은 철회한다.
2. SRCIST 가중치: 순위로 5등분한 뒤 역빈도를 사용하여 각 구간 크기가 거의 동일하다. 따라서 가중치는 거의 1이며 희귀 수명구간 강조라고 볼 수 없다.
3. 최종 재학습: SRCIST/MCST/EDIST runner의 evaluate는 development를 train과 validation 양쪽으로 전달한다. 고정 epoch로 기본 신경망을 재학습하는 것과 달리, 보정/gate/DRO 위험까지 동일 훈련 데이터에서 재추정하게 된다. 외부 라벨 누출은 아니지만 held-out validation에서 결정한 규칙을 배포한다는 설명과 다르다.
4. MCST/EDIST는 내부 gate에 최악 위험을 사용하지만 grid 선택은 여전히 반환 validation_mse의 평균으로 한다. 전체 선택 절차가 최악 환경 위험을 최소화하는 것은 아니다.
5. EDIST progress_index=-1 처리: _latent_coordinates는 0 이상의 열 번호와 -1을 직접 비교하므로 Axial의 마지막 progress 열을 제거하지 않는다. 실제로는 수명 진행도가 환경 군집에 포함된다.
6. CIST mean loss는 zero-noise 출력을 학습하지만 validation과 추론은 noise에 대한 평균을 사용한다. 비선형 모델에서는 두 값이 같지 않다. 평균 R2를 목표로 삼는 손실과 추론 대상 사이에 불일치가 있다.
7. CIST의 양의 rate는 전체 예측 단조성을 보장하지 않는다. 문맥, anchor, 입력 의존 gate와 direct 경로도 변하며 midpoint 적분 근사까지 포함된다. 전체 단조성 또는 물리 제약을 주장하려면 별도 근거가 필요하다.
8. 공통 CIST가 이전 실험 도중 결정론적 적분으로 변경됐다. 초기 CIST와 후속 변형은 공통 기반이 고정된 제거 실험이 아니다.
9. Axial validation: CIST 계열은 last_per_group으로 마지막 행만 선택한다. Engression 비교 runner는 해당 축약 없이 validation을 사용한다. 최종 truth 배열은 동일하지만 모델 선택 조건은 동일하지 않다.
10. EDIST는 leave-one-regime-out CVaR 학습, nuisance head, seed 기울기 페널티를 구현하지 않았다. 현재 구현은 군집 기반 1회 재가중과 혼합이다.

## 저장 예측 재집계

완료된 CIST/SRCIST/MCST/RCIST와 4모형 비교의 truth 배열은 모든 설정에서 정확히 일치했다. 예측의 단위 ID는 저장되지 않아 truth 일치만으로 ID 정렬까지 독립 입증하지는 못한다.

| 모델 | 설정 평균 R2 | 설정 내 TSS 가중 R2 | 단순 연결 pooled R2 | 앙상블 이득 |
|---|---:|---:|---:|---:|
| MLP | .432483 | .718306 | .753532 | .007721 |
| Engression | .474173 | .734563 | .767755 | .117769 |
| PP-X | .488820 | .728669 | .762598 | .010301 |
| CDCR-PPX | .488372 | .728578 | .762519 | .002575 |
| CIST | .479663 | .726677 | .760855 | .058493 |
| SRCIST | .479234 | .730436 | .764145 | .050936 |
| MCST | .474441 | .729029 | .762914 | .049557 |
| RCIST | -.082454 | .460069 | .527586 | .022851 |

설정 내 TSS 가중 R2 = 1 - sum(SSE_d)/sum(TSS_d). 단순 pooled R2는 모든 설정의 y를 연결하고 전체 평균으로 TSS를 계산했다. 서로 다른 출력 스케일을 섞으므로 둘 다 설명용이며 단일 공정 성능 기준으로 채택하지 않는다. 앙상블 이득은 설정별 앙상블 R2 - 개별 seed R2 평균의 설정 평균이다.

행 수는 Axial 각 100, MATWI 168, Misata 1298이다. Misata는 1766행 중 약 73.5%이며 설정 내 TSS 합의 67.27%다. 5설정은 5개의 독립 코호트가 아니라 Axial 3설정과 MATWI, Misata다.

Engression의 평균 seed R2는 .356404, PP-X는 .478519다. 그러나 Engression 앙상블 이득은 .117769로 PP-X의 .010301보다 크다. seed R2 SD만으로 오차의 상보성이나 예측 분산을 설명할 수 없다. GCS의 외부 평균 .489340과 seed SD .003413은 저장 요약값이며, 후처리의 seed 간 결합 여부를 이번 감사에서는 추가 확인하지 않았으므로 독립 학습 안정성으로 단정하지 않는다.

4P_1F에서 CIST의 평균 잔차/target SD는 -.003인데 R2는 .496 수준이다. 단순 절편 보정만으로 Engression과의 차이를 메우기는 어렵다. RCIST의 Axial 표준화 편향은 -.802, -.862, -1.073으로 큰 하향 편향이 관측된다. 이것만으로 각 구성요소의 인과적 책임을 나눌 수는 없다.

## 다음 작업 순서

1. 기본 모델/전처리/예측 집계/validation 행 선택을 고정한다. source 장비 ID를 유지하여 최종 gate는 out-of-fold 예측에서만 정하고 재학습 시 고정한다.
2. PP-X와 CIST 기준을 동일 조건으로 확립하고 단일 seed 성능과 동일 개수 독립 모델 앙상블 성능을 함께 비교한다. 초기 9설정 보존도 별도로 평가한다.
3. 첫 구조 실험은 CIST의 zero-noise MSE를 실제 예측 평균 MSE로 맞추는 것 하나로 제한한다. 개선은 가설이며 아직 실행하지 않았다.
4. 다음 제거 실험은 적분 경로/직접 경로와 제약 강도를 분리한다. 현재 전체 모델에 없는 단조성 보장을 전제로 해석하지 않는다.
5. source 장비/실제 운전조건 제외 검증에서 개선이 반복될 때만 조건별 기울기 보정 구조를 설계한다. 자동 군집이 실제 운전환경이라는 가정을 검증 없이 사용하지 않는다.
6. 평가에는 설정 평균, 코호트 균등 평균, 최악 설정, 개별 seed 하한, 장비 단위 불확실성을 포함한다. 외부 5설정은 개발 벤치마크로 표시하고 새 코호트로 최종 확인한다.

## 주요 근거 파일

- experiments/cist_ppx_external_development.py: last_per_group, select, evaluate, metric
- experiments/external_cohort_four_model_comparison.py: axial_settings, run_engression, summarize
- experiments/srcist_ppx_external_development.py: evaluate
- experiments/mcst_ppx_external_development.py: select, evaluate
- experiments/edist_ppx_external_development.py: select, evaluate
- src/pp_extrapolation/innovation_slope_transport.py: mean loss와 predict 경로
- 대화에 남은 innovation_source_robust_transport.py, innovation_minimax_consensus_transport.py, environment_dro_slope_transport.py 생성 패치
- results/*_external_development_v1/predictions.npz 및 external_cohort_four_model_comparison_v1/predictions.npz
