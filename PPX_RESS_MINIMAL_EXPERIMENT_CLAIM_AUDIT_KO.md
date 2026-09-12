# PP-X: RESS 목표 최소 추가실험 주장·근거 종합 분석

작성일: 2026-09-13

## 1. 결론

현재 PP-X는 새로운 생성 모델의 성공을 기다리지 않고 PHM 방법론 논문을 작성할 근거가 있다. RESS를 목표로 한 중심 메시지는 **구조적 프라이어를 사용하는 외삽에서의 일관된 예측 성능**으로 잡는다. 게재 가능성이나 안전성 보장이 확보됐다는 뜻은 아니다.

현재 직접 뒷받침되는 결과 문장:

> PP-X는 9개 후향적 외삽 평가 설정에서 높은 예측 성능을 일관되게 유지했으며, 강화된 경쟁모델 비교에서 8개 설정의 최강 비교군보다 높은 R²를 보였다. 이 평가 범위에서 데이터셋 간 성능 편차와 음의 R² 발생도 비교군보다 작았다.

연구 동기·설계 가설 문장:

> 정당화되는 구조적 프라이어를 보존하면서 데이터 기반 보정을 학습하면, 관측 범위 밖에서 발생하는 성능 저하와 변동을 줄일 수 있다는 가설을 평가한다.

두 문장을 합쳐 **“프라이어가 참이기 때문에 안정적이라는 인과적 명제를 입증했다”**고 쓰지 않는다. 현재 기록은 프라이어의 참·거짓을 독립적으로 확정한 실험이 아니다.

권장 영문 결과 문장:

> Across nine retrospective extrapolation settings, PP-X maintained positive ensemble R² in every setting, with a macro-average R² of 0.807 and a minimum of 0.466. It exceeded the strongest evaluated comparator in eight settings and exhibited lower observed cross-setting dispersion. These results support consistent performance within the evaluated benchmark, rather than universal robustness or causal identification of prior validity.

## 2. 이번 작업의 범위

- 새 모델 학습, 예측 재생성, 검정 재실행은 하지 않았다.
- 기존 최종 모델 정의, 논문 evidence registry, 동일 후보예산 비교, 구성요소 분석, 안정성 JSON, DS03 및 구버전 외부 평가 기록을 종합했다.
- 수치는 기존 산출물에서 가져온 것이며, 아래 분석은 근거의 해석과 주장 범위 감사다.
- 모든 실험 폴더의 전수 원예측 감사나 모든 역사적 코호트의 단일 재집계를 완료했다는 뜻은 아니다. 원예측의 행 정렬·모델 체크포인트·프로토콜 시점도 이번에 재검증하지 않았다.
- 결론의 모집단은 **기존 9개 평가 설정**이다. 개발 과정에서 사용한 설정이라는 지위를 유지한다.
- 현재 모델은 최종 논문용 PP-X framework다. 초기 PP, PP-X v1, CCMR, CIST, 적분 잔차 후보를 한 버전으로 합치지 않는다.

## 3. 가장 강한 정량 근거: 성능 수준과 변동을 함께 보기

출처: `results/ppx_domain_stability_v1/results.json`. 설정별 5-seed ensemble R²의 분포다. 서로 다른 목표값을 전부 연결한 pooled R²가 아니다.

| 모델 | 설정 평균 R² | 설정 간 SD | 설정 간 MAD | 최저 설정 R² | R² > 0 |
|---|---:|---:|---:|---:|---:|
| PP-X | 0.8071 | 0.1737 | 0.0611 | 0.4657 | 9/9 |
| Engression | 0.2567 | 0.9486 | 0.2797 | -1.5804 | 7/9 |
| plain MLP | 0.0934 | 1.0069 | 0.5541 | -2.1396 | 6/9 |
| FT-Transformer | 0.1151 | 1.0050 | 0.3657 | -2.0095 | 7/9 |
| GroupDRO | 0.0107 | 0.9975 | 0.6189 | -2.0571 | 5/9 |
| V-REx | 0.0712 | 1.0075 | 0.5349 | -2.1876 | 6/9 |
| monotone NN | 0.0538 | 1.0568 | 0.5414 | -2.3507 | 6/9 |
| linear-tail RBF | -0.0846 | 1.4730 | 0.1156 | -2.6486 | 7/9 |
| SVGP | -1.6394 | 4.2574 | 0.5087 | -12.5841 | 5/9 |

해석:

- PP-X는 낮은 성능으로 일정한 모델이 아니라, 높은 평균과 높은 최저 성능을 함께 보였다.
- PP-X의 설정 간 SD는 Engression보다 작고, 저장된 SD 비율은 5.46이다. 이를 “미래 실패 위험 5.46배 감소”로 바꾸지 않는다.
- SD만이 아니라 MAD, 최저 성능, 양의 R² 설정 수를 같이 보여준다. R²의 음의 꼬리와 데이터셋별 목표 분산 차이가 SD에 영향을 주기 때문이다.
- 양의 R²는 해당 평가 목표의 평균 예측 기준보다 낫다는 뜻이지, 현장 사용 가능성 또는 안전 기준 통과를 뜻하지 않는다.
- 이 표로 시드 간 변동이나 모든 개체에서의 안정성까지 입증할 수 없다.

### 통계적 경계

- Engression 대비 paired exact MAD 검정: 양측 p=0.0390625.
- 8개 비교를 포함한 Holm 보정: Engression 비교 q=0.2734375. 모든 MAD 비교가 보정 후 비유의다.
- 저장된 bootstrap의 `SD(Engression)-SD(PP-X)` 95% 구간은 [0.0051, 0.9899]이다. 이 결과가 다중비교 비유의나 개발 설정 재사용의 한계를 없애지는 않는다.
- 따라서 **“관찰된 편차가 작다”는 강하게**, **“모든 모델보다 통계적으로 안정적이다”는 쓰지 않는다.**
- 9개 setting은 모두 독립적인 산업 도메인 9개가 아니다. 배터리 설정이 다수이며 데이터 계열의 유사성을 밝힌다.

## 4. 정확도 비교: 미세 우위와 큰 개선 구분

출처: `FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md`.

| 설정 | PP-X R² | 최강 비교군 | 비교군 R² | 판단 |
|---|---:|---|---:|---|
| HUST | 0.958 | GroupDRO | 0.955 | 작은 우위 |
| Virkler | 0.888 | FT-Transformer | 0.890 | 작은 열세 |
| NASA battery | 0.584 | Engression | 0.583 | 사실상 근접 |
| Sunwoda | 0.939 | linear-tail RBF | 0.838 | 뚜렷한 점추정 개선 |
| RWTH | 0.878 | linear-tail RBF | 0.732 | 뚜렷한 점추정 개선 |
| MICH | 0.751 | monotone NN | -0.686 | 큰 성능 붕괴 방지 신호 |
| MATR2019 | 0.466 | FT-Transformer | 0.342 | 점추정 개선 |
| MATR-b2 | 0.862 | plain MLP | 0.813 | 점추정 개선 |
| N-CMAPSS DS02 | 0.937 | Engression | 0.932 | 작은 우위, 개체 이질성 |

- 전체 승수 8/9, 기존 dataset sign test 양측 p=0.0391. 개발 후향적 분석이며 독립 확증의 성공 확률이 아니다.
- 후보 수 30개와 refit 시드 5개는 8개 경쟁모델에 동일하게 적용됐다. PP-X의 누적 구조개발 비용·후보 수·시간까지 동일한 것은 아니다.
- 최강 비교군은 test 성능을 보고 고른 비교용 상한이다. 배포 가능한 사전 선택 모델이나 단일 사전 지정 검정으로 기술하지 않는다.
- 최신 동일 후보예산 비교의 unit-level BH 유의 설정은 Sunwoda, RWTH, MICH, MATR-b2다. 기존 mixed/row-aligned comparator 분석의 “2개 유의”와는 비교군이 다르므로 합치지 않는다.
- N-CMAPSS는 pooled 우위에도 unit 평균 log-RMSE 효과가 음수다. “모든 개체에서도 덜 흔들린다”는 근거로 사용할 수 없다.

## 5. 프라이어 적용 조건: 사후 성공 분류를 피하기

아래는 도메인 가정을 서술하기 위한 정리이지, 각 데이터셋에서 프라이어가 참임을 검정한 결과가 아니다. 가정이 물리적으로 그럴듯한 것, source 데이터에서 지지되는 것, 새로운 개체로 전이되는 것은 서로 다르다.

| 설정군 | 설명할 프라이어·조건 | 반드시 밝힐 한계 |
|---|---|---|
| Sunwoda/RWTH/MICH | 경계 기반 quotient/잔차 경로, 인과적 상태 이력 | 고정 bound와 dual-scale의 효과가 다름. 알려진 경계만으로 개체 간 수명 관계가 같아지지 않음 |
| HUST/MATR-b2 | 상태-수명 관계와 source 근거에 따른 transport | transport의 전이 가능성이 추가 가정. 경계의 존재만으로 정당화할 수 없음 |
| Virkler | 균열 진행 좌표와 잔차의 외삽 제한 | 기존 계약·종료 기준을 정확히 서술. 성공 점수로 prior validity를 정의하지 않음 |
| NASA/MATR2019 | 관측 건강도·용량과 인과적 이력을 이용한 수명 외삽 | 마지막 기록까지 남은 시간과 실제 물리적 EOL의 차이, 고정 prefix 미래예측과의 차이 |
| N-CMAPSS DS02 | 작동 조건·이력에 기반한 prior-residual 외삽 | TRA는 물리적 열화 시간이 아니며 명시적 공통 EOL 좌표도 아님. 경계 프라이어가 참인 집단에 자동 편입하지 않음 |

최소 서술 규칙:

1. 관측 가능한 경계, 허용 이력, regime, source 지원을 먼저 정의한다.
2. test 성능으로 “prior-valid”와 “prior-invalid”를 나누지 않는다.
3. source/validation 승인도 참인 프라이어를 보증하는 것이 아니라 적용을 위한 운영 규칙으로 기술한다.
4. 조건별 하위집단 수치를 새로 내면 사후 탐색 분석임을 명시한다.

### 이미 존재하는 반증성 결과

`PPX_ENGRESSION_WINLOSS_DESCRIPTORS_KO.md`에서 boundary informativeness와 PP-X의 Engression 대비 이득 간 상관은 rho=-0.250, p=0.5165였다. 전 9개 설정에서 단순한 “경계 정보가 강할수록 이득이 커진다”는 관계는 지지되지 않았다.

따라서 **“프라이어 타당성에 따라 이득이 증가하는 법칙을 이미 확인했다”는 문장은 금지**한다. n=9의 대용 지표 분석이므로 프라이어 기반 접근 자체의 반증도 아니다.

## 6. 구조적 설명을 뒷받침하는 기존 ablation

출처: `results/ppx_final_ablation_statistics_v1/REPORT_KO.md`.

| 요소 | 근거 | 주장 가능한 범위 |
|---|---|---|
| 비선형 잔차 | Sunwoda/RWTH/MICH에서 unit 검정 BH 유의 | prior-only에 데이터 기반 보정이 도움이 되는 설정이 있음 |
| frozen affine | 세 설정 모두 점추정 양수, CI는 0 포함 | 설계상의 기준 경로. 독립적인 안정화 효과 입증은 아님 |
| fixed bound | Sunwoda +0.221 R², MICH -0.291 | 항상 적용하는 제한은 정당화되지 않음 |
| dual scale | MICH +0.283, RWTH -0.037 | 조건별 executor가 필요하다는 표본 내 근거 |
| regime transport | HUST +0.128, MATR-b2 +0.187 | 두 설정에서의 개선 근거 |
| history | NASA +0.012, N-CMAPSS +0.009 | 작은 방향성 효과, 개체 수 부족 |

주의: MICH core ablation의 on R²=0.468과 최종 dual-scale R²=0.751은 다른 arm이다. 이를 같은 최종 모델의 직접 비교처럼 합치지 않는다.

현재 ablation은 주로 정확도 효과다. 이것만으로 **프라이어 보존이 시드/개체/도메인 분산 감소의 원인**임을 입증했다고 쓰지 않는다. RESS 원고에서는 “설계를 뒷받침하는 조건부 기전 증거”로 사용한다.

## 7. 외부 평가와 버전별 반례

| 증거 | 결과 | 원고에서의 사용 |
|---|---|---|
| DS03 동결 prospective PP-X | fallback 0.882, Engression 0.901; PP-X 후보 중 fallback이 최고 | 한 코호트에서 올바른 후보 선택. 예측 우월성 실패도 함께 보고 |
| Stanford 초기 PP-X v1 | PP-X -0.228, matched MLP 0.034 | 구버전의 외부 실패. 최신 PP-X의 실패와 동일시하지 않되 프로그램의 한계로 공개 |
| NASA second/UL-PUR/SNL/CALB/Tongji v1 큐 | 미채점: 부적격/표본 부족/외삽 구간 부재 | 정확도 성공 또는 낮은 위험으로 계산하지 않음 |
| FEMTO/XJTU/milling | 최종 registry의 limitation/development tier | 버전·후처리·개발 지위를 분리하고 main 승수에 합치지 않음 |
| 이후 CIST/적분 보정 후보 | 별도 구조의 개발 실험 | 최종 PP-X 성능으로 승격하거나 PP-X 자체의 직접 실패로 합치지 않음 |

DS03는 N-CMAPSS 계열이므로 새로운 산업 도메인의 독립 확증으로 과장하지 않는다. 문서의 route-selection success를 “선택정책이 미래에 일반화함을 확정”으로 확대하지 않는다.

구버전 실패를 나중에 “프라이어가 틀린 데이터였으므로 제외”한다고만 설명하면 사후 면책이 된다. 적용 조건, 결과 열람 시점, 버전 차이를 함께 밝힌다.

현재 검토한 기록만으로 모든 역사적 코호트의 동일 버전 통합 안정성을 확정할 수 없다. 그런 주장은 이 보고서에서 의도적으로 보류한다.

## 8. 원고용 주장 등급

| 문장 | 판정 |
|---|---|
| 9개 평가 설정 모두에서 양의 R²와 높은 최저 성능을 유지했다 | 직접 사용 가능 |
| 해당 설정에서 비교군보다 작은 성능 편차를 보였다 | 기술적 결과로 사용 가능 |
| 조건별 prior-residual executor의 이득과 반례를 확인했다 | ablation 범위를 밝히고 사용 가능 |
| 정당화된 프라이어를 활용하는 외삽을 대상으로 한다 | 연구 범위·설계 원칙으로 사용 가능 |
| 프라이어가 참일 때 안정성이 높아진다는 법칙을 입증했다 | 불가 |
| 프라이어 보존 자체가 안정성 향상의 원인임을 입증했다 | 현재 증거로 불가 |
| 모든 코호트·시드·개체에서 덜 흔들린다 | 불가 |
| 모든 비교군보다 분산이 유의하게 작다 | Holm 보정 결과와 불일치 |
| PP-X와 비교군의 총 개발/계산 예산이 같다 | 불가 |
| 안전한 RUL, 실패 방지, 보편적 Engression 우월 | 불가 |

## 9. 실험을 최소화한 RESS 원고 구성

제목 작업안:

> Consistent RUL Extrapolation with Contract-Conditioned Prior-Residual Learning

이 제목의 consistent는 통계적 추정량 일치성 정리가 아니라 경험적 성능 유지라는 뜻으로 초록에서 명시한다.

1. 문제: 관측 열화/작동 범위 밖에서 데이터 기반 수명 예측이 불안정해질 수 있다.
2. 방법: 허용 프라이어, prior-residual core, validation 승인 executor와 fallback을 정의한다.
3. 주 결과: 전체 9개 setting의 정확도·최저 성능·양의 R² 비율·MAD를 함께 제시한다.
4. 기전: 기존 component ablation과 모듈 적용의 반례를 제시한다.
5. 한계: DS03, 구버전 외부 실패, 후향적 개발 지위, 목표값 정의와 개체 수를 제시한다.

필수 표/그림은 기존 산출물로 구성할 수 있다:

- 전체 모델의 setting별 R² 표와 점도표. 평균과 SD 하나만 남기지 않는다.
- 평균 성능-설정 간 MAD의 이차원 비교. 좋은 성능과 낮은 변동을 동시에 읽게 한다.
- 조건별 component on/off 표. 부정적 효과를 생략하지 않는다.
- main/prospective/legacy failure/infeasible의 evidence registry.

현재 주장을 위 범위로 제한하면 새 대규모 학습을 원고 작성의 선행조건으로 둘 필요는 없다. 단, “프라이어가 타당하기 때문에 덜 흔들린다”를 결론으로 고집하면 독립적인 타당성 정의와 통제 실험이 추가로 필요하다.

## 10. 실험 없이 남은 편집·재현성 쟁점

- 고정 프로토콜은 2% validation gain을 명시하지만 최종 모델 설명에는 unit win 60%, worst-unit ratio 1.10도 등장한다. 이번에는 구현을 재감사하지 않았으므로 Methods 작성 전 어느 버전의 규칙인지 대조해야 한다. 이번 분석에서 임의로 통일하지 않았다.
- 최초 고정 시점과 과거 데이터별 최종 route 생성 시점을 구분한다. 선택 API의 존재만으로 과거 결과 전체가 그 API로 사전 생성됐다고 주장하지 않는다.
- 본 분석의 안정성 수치는 5-seed 최종 결과다. 최근 3-seed 적분 후보 비교의 PP-X 참조 수치로 교체하지 않는다.
- 용량 꼬리의 현재 상태 기반 RUL 추정과 고정 prefix에서 미래 전체를 예측하는 문제를 구분한다.
- 기록 종료까지의 RUL과 물리적 고장/EOL을 동일시하지 않는다.
- 새로운 위험 분석이나 정비 시뮬레이션을 수행하지 않았으므로 R² 개선을 안전·정비 비용 절감으로 환산하지 않는다.

## 11. 최종 판단

**현재 가장 경제적인 전략은 PP-X를 고정하고, 관찰된 높은 성능 유지와 조건부 구성요소 근거를 중심으로 원고를 작성하는 것이다.**

주장 강도는 “평가 범위에서 일관된 성능”에 둔다. “정당한 프라이어”는 적용 범위와 설계 동기로 사용하되, 점수로 참·거짓을 정의하거나 이미 입증된 안정성 법칙으로 바꾸지 않는다. 기존 자료만으로도 원고 작성은 진행할 수 있지만, RESS 게재 또는 심사 중 추가 실험 불필요를 보장할 수는 없다.

## 12. 근거 파일

- [최종 모델 정의](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/PPX_FINAL_PAPER_MODEL_KO.md)
- [통합 evidence registry](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md)
- [고정 프로토콜](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md)
- [동일 후보예산 비교](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md)
- [안정성 분석](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/PPX_DOMAIN_STABILITY_RESULTS_KO.md)
- [안정성 수치 JSON](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/results/ppx_domain_stability_v1/results.json)
- [구성요소 분석](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/results/ppx_final_ablation_statistics_v1/REPORT_KO.md)
- [승패 설명변수 분석](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/PPX_ENGRESSION_WINLOSS_DESCRIPTORS_KO.md)
- [DS03 외부 결과](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/NC_MAPSS_DS03_PPX_PROSPECTIVE_RESULTS_KO.md)
- [초기 v1 외부 결과](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/PPX_V1_POST_PRIMARY_COHORT_RESULTS_KO.md)
- [코호트 재사용 감사](/Users/baghyeongbae/Desktop/연구/pp-extrapolation/COHORT_REUSE_AUDIT_KO.md)

문서에서 인용한 기존 결과는 원저자 기록과 저장된 요약 통계에 대한 분석이다. 이번 작업에서 새 통계 검정을 실행했다는 의미는 아니다.
