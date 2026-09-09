# 소형 비배터리 외부 cohort 탐색 감사

결과가 좋아 보이는 자료만 골라 성공으로 선언하지 않도록, 파일을 열기 전에 두 crack cohort의 split·최소 표본 기준을 각각 커밋으로 고정했다.

| 후보 | 크기 | 독립 test 구성 | 판정 | 이유 |
|---|---:|---:|---|---|
| SJTU fatigue-crack workbook | 43 KB | 구성 불가 | inconclusive | 4개 sheet가 4개 specimen이 아니라 `E_0_175_1`, `C_15_20_1` 두 specimen의 서로 다른 분석 section이었다. |
| NASA PHM 2019 lap joint | 1.89 MB | T7, T8 | inconclusive | train-only 경계 3.64 mm 전에 각 test specimen이 5개 관측만 제공한다. 최초 crossing을 포함해도 6개라 사전 최소 10개 규칙을 통과하지 못한다. |
| C-MAPSS FD001/FD003 | 약 43 MB(로컬 6개 txt 합계) | 공식 test engine 각 100대 | PP-level 완료 | 배터리가 아닌 turbofan이며 충분히 작다. 데이터 자체는 과거 PAE에서 사용했으므로 dataset-level untouched라고 부르지 않는다. |

SJTU와 NASA crack은 성능 실패가 아니라 **식별 가능한 평가 표본 부족**이다. 최소 관측 규칙을 결과를 본 뒤 낮추지 않았다. 두 자료는 향후 고빈도 crack-length truth 또는 더 많은 독립 specimen이 확보될 때 다시 사용할 수 있다.

C-MAPSS의 첫 동결 PP 실행은 uncapped physical RUL에서 양의 R²를 냈지만 MLP에 졌다. 이후 원인을 운전조건과 열화가 섞인 표현으로 진단하고, 학습 engine만으로 운전조건별 정규화와 초기상태 대비 센서 변화를 계산하는 표현을 개발했다. 이 성공 결과는 별도 문서에 **posthoc repair**로 표시한다.

## 추가 실행: axial fan

Mendeley axial-fan 데이터는 12개 텍스트 파일 합계 9 MB 미만, 설정별 train/test fan 각 100대로 크기와 표본 조건을 모두 만족했다. 세 outcome-sealed 설정에 고정 PP를 적용했지만 PP pooled R² -0.746, matched MLP -0.455로 확증에 실패했다. 이 자료는 판정 불가가 아니라 충분한 표본에서 얻은 외부 negative cohort이며 `AXIAL_FAN_UNTOUCHED_RESULTS_KO.md`에 상세히 기록했다.

공식 결과를 확인한 뒤 endpoint 분포 불일치를 수정한 개발 실험에서는 PP
pooled R²가 0.591로 회복됐고 matched MLP 0.502를 앞섰다. 이 결과는
`AXIAL_FAN_PP_REPAIR_KO.md`에 기록했으며 untouched 성공으로 소급하지 않는다.

## 후속 저용량 후보 감사

| 후보 | 크기/독립 단위 | 판정 | 핵심 이유 |
|---|---:|---|---|
| MEMSS Fatigue | 4.9 KB / 21 | inconclusive | 사전 test 중 경계 도달 2개 |
| GaAs laser | 1.9 KB / 15 | inconclusive | 공식 10% 경계 도달 3개 |
| MATWI labels | 535 KB / 17 | 확증 실패 | PP -0.137, MLP -0.151; 절삭 context 없는 수명척도 이동 |
| Misata machine | 526 KB / 100 | PP 우월성 실패 | PP 0.833, MLP 0.854; 둘 다 양의 절대 성능 |
| Device-B | 4.0 KB / 34 | inconclusive | 충분한 pre-failure history를 가진 경계 도달 장치 4개 |

이 감사는 작은 파일 크기와 유효한 외삽 표본 수가 별개임을 보여준다. 공통
경계를 가진 고전 accelerated-degradation 자료도 빠른 경계 도달 또는 검열
때문에 독립 확증에 필요한 history와 unit 수를 동시에 충족하지 못할 수 있다.
