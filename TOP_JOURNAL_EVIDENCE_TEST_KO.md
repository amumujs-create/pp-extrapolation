# PP 상위저널 증거 게이트 테스트

## 종합 판정

총 11개 게이트 중 **통과 5, 부분통과 3, 실패 3**이다. 방법·성능·다중도메인 통계는 강해졌지만, Oxford outcome-held-out cohort까지 실행했지만 사전 성공 기준을 통과하지 못했으므로 최상위 수준의 확증 상태는 아니다.

| 게이트 | 판정 | 핵심 결과 |
|---|---|---|
| 강한 외삽 경쟁모델 | PASS | 양의 8개 설정 모두 PP pooled R²가 공식 Engression보다 높음 |
| 계층 효과크기 | PASS | 상대 RMSE -24.0%, 95% CI [-40.9%, -5.5%] |
| leave-one-dataset-out | PASS | 8개 데이터셋 어느 하나를 제외해도 CI 상한 < 0 |
| causal feature ablation | PASS | NASA와 N-CMAPSS 모두 adaptive PP가 모든 고정 preset보다 높음 |
| label-noise 강건성 | PASS | NASA 0–20% train-label noise에서 R² 0.583–0.594 |
| uncertainty localization | PARTIAL | HUST error 상관 ρ=0.225이나 상수 shrinkage 대비 추가 이득 0.47% |
| applicability certificate | PARTIAL | 6개 중 5개 판별, false accept 0, false reject 1; 사후 임계값 |
| 계산비용 보고 | PARTIAL | PP profile 완료, 경쟁모델 동일 하드웨어 profile 미완료 |
| NASA unit 일관성 | FAIL | Engression 대비 2/4 battery 승리 |
| N-CMAPSS 독립 unit 수 | FAIL | 평가점 2개 이상인 test engine이 2개뿐 |
| 최종 모델 새 cohort | FAIL | Oxford outcome-held-out 실행 완료; PP가 경쟁모델보다 우수했지만 R² -0.119로 사전 성공 기준 실패 |

## 새 ablation

### N-CMAPSS

| 고정 구조 | Ensemble pooled R² |
|---|---:|
| Basic history | 0.928 |
| Moments history | 0.929 |
| Multiscale history | 0.924 |
| **Validation-adaptive PP** | **0.937** |

큰 feature set 하나가 우연히 성능을 올린 결과가 아니다. seed별 validation이 history representation과 regularizer를 선택할 때 가장 높았다. 다만 per-seed 선택은 고정 단일 configuration보다 탐색 예산이 크므로 논문에 후보 수를 명시해야 한다.

### NASA

| 구조 | Ensemble pooled R² | Unit-macro R² |
|---|---:|---:|
| Scalar-health regime PP | 0.571 | 0.616 |
| Fixed short history | 0.572 | 0.657 |
| Fixed multiscale | 0.411 | 0.283 |
| Fixed moments | 0.401 | 0.268 |
| **Validation-adaptive causal PP** | **0.584** | **0.661** |

NASA에서도 무조건 긴 history를 쓰면 크게 악화한다. validation 기반으로 short/multiscale/moments를 fold별 선택하는 실행 규칙이 성능에 필요하다.

## Label-noise 강건성

NASA 최종 구조와 clean-validation 선택을 고정한 뒤 train target 표준편차의 5%, 10%, 20% Gaussian noise를 추가했다.

| Train-label noise | Pooled R² |
|---:|---:|
| 0% | 0.584 |
| 5% | 0.583 |
| 10% | 0.583 |
| 20% | 0.594 |

20%에서 오히려 높아진 것은 noise가 regularization처럼 작용한 사후 관측이며 성능 향상 주장으로 사용하지 않는다. 중요한 결과는 10%까지 성능 하락이 0.001 미만이라는 점이다.

## 계산비용

Apple arm CPU, PyTorch 2.14, CPU thread 2개에서 hyperparameter search와 데이터 로딩을 제외한 대표 단일 fit을 측정했다.

| 설정 | 파라미터 | Train rows | 학습시간 | 추론 배치 | μs/row |
|---|---:|---:|---:|---:|---:|
| NASA 대표 fold | 523 | 121 | 0.51 s | 56 | 1.97 |
| N-CMAPSS seed42 | 7,184 | 5,239 | 2.52 s | 13,500 | 0.45 |

프로세스 peak RSS는 약 1.78 GiB였지만 Python·PyTorch·전체 데이터 로딩을 포함하므로 모델 단독 메모리로 해석하지 않는다. Engression, FT-Transformer, TabPFN을 같은 프로파일러로 측정해야 계산 효율 게이트가 완전히 통과한다.

## 제출 전에 해결해야 할 세 항목

1. 최종 causal modular PP와 선택 규칙을 commit으로 고정한 뒤 새 battery/engine cohort에서 한 번 평가한다.
2. NASA형 독립 battery와 N-CMAPSS형 test engine 수를 늘린다. seed 추가로 물리적 표본 부족을 대신하지 않는다.
3. uncertainty head가 상수 shrinkage보다 실제로 더 나은 risk–coverage를 보이는 새 cohort 결과를 확보한다.

현재 결과는 RESS/MSSP/EAAI 계열에 제출할 수 있는 강한 retrospective 방법론 증거다. 더 높은 일반화 주장은 위 세 항목 중 첫 번째가 완료된 뒤 사용하는 것이 안전하다.

## 재현 위치

- 자동 게이트: `experiments/top_journal_evidence_audit.py`
- 게이트 결과: `results/top_journal_evidence_audit_v1/results.json`
- N-CMAPSS ablation: `results/ncmapss_ablation_*_v1/`
- NASA ablation: `results/nasa_ablation_*_v1/`
- noise 실험: `experiments/nasa_causal_robustness.py`, `results/nasa_causal_robustness_v1/`
- compute profile: `experiments/profile_final_pp.py`, `results/final_pp_compute_profile_v1/`
