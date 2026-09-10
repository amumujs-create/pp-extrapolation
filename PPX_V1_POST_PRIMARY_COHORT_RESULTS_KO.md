# PP-X v1 1차 이후 외부 고호트 결과

기록: 박진서. 2026-09-10.

1차 결과는 주표 9셋 개발 포트폴리오와 이미 끝난 고호트(MATR 2019/batch2, Misata, Oxford 등)다. 이 문서는 그 이후, **개선 전 PP-X v1.0**을 새 미개봉 아카이브에 그대로 적용한 큐다. Oxford 스케일 패치는 쓰지 않았다. 게이트를 무시하고 억지 점수를 낸 실행은 없다.

모델은 모든 실행에서 같다. affine prior + residual 코어, `select_residual_executor`로 unbounded(`residual_decay=0`) 대 bounded(`residual_decay=0.3`), 폭 `{16,32}`, 학습률 `{5e-4,1e-3}`, seed 42–46, matched MLP. 확증 성공은 ensemble pooled R² > 0 이고 PP-X RMSE가 MLP보다 낮을 때만이다.

## 총평

**새 확증 성공은 없다.** 앞선 다섯 아카이브는 적격 또는 1D 헐에서 멈춰 점수가 없다. 여섯 번째 Stanford는 계약이 통과해 채점까지 갔고, **PP-X v1이 matched MLP에 진 음수 확증**이다. 실패한 고호트의 컷·분할·Ref 제외를 사후 변경하지 않는다.

| 순서 | 고호트 | 프로토콜 | 판정 | PP-X / MLP pooled R² |
|---|---|---|---|---|
| 1 | NASA PCoE second (B0029–B0056) | `NASA_PCOE_SECOND_COHORT_V1` | inconclusive | 미채점 |
| 2 | UL-PUR | `UL_PUR_PPX_V1` | inconclusive | 미채점 |
| 3 | SNL | `SNL_PPX_V1` | infeasible | 미채점 |
| 4 | CALB | `CALB_PPX_V1` | inconclusive | 미채점 |
| 5 | Tongji | `TONGJI_PPX_V1` | infeasible | 미채점 |
| 6 | Stanford | `STANFORD_PPX_V1` | 확증 완료, 실패 | **−0.228 / 0.034** |

## 게이트에서 멈춘 다섯 고호트

점수가 없는 것은 스크립트 오류가 아니다. 사전 적격·헐 규칙이 채점 무대를 열지 않았다.

### NASA PCoE second

고정 11개 MAT만 파싱했다. 방전 40회 이상이고 첫 5사이클 중앙값의 70%를 교차해야 적격이다. 11셀 모두 실패했고 ID를 대체하지 않았다. B0029–B0048은 교차가 없고, B0054–B0056은 교차 인덱스가 0이라 이력 부족이다.

- 결과: `results/nasa_pcoe_second_ppx_v1/results.json`
- 코드: `experiments/nasa_pcoe_second_ppx_v1.py`

### UL-PUR

BatteryLife processed `UL_PUR.zip`(SHA-256 `da2d5815…4f02f2`). 셀 2개, 80% 교차 적격 0. 표본 부족으로 inconclusive. 이 판정 때문에만 사전 등록한 SNL로 넘어갔다.

- 결과: `results/ul_pur_ppx_v1/results.json`
- 코드: `experiments/ul_pur_ppx_v1.py`

### SNL

`SNL.zip`(SHA-256 `bb2bff21…2892d`). 61셀 중 적격 34. 사전순 60/20/20이 LFP(~1 Ah)와 NCA/NMC(~3 Ah)를 섞는다. 원 Ah 학습 최솟값 0.829 아래로 검증·테스트 행이 없어 infeasible. 점수를 본 뒤 정규화 건강도로 SNL을 재실행하지 않았다.

- 결과: `results/snl_ppx_v1/results.json`
- 코드: `experiments/snl_ppx_v1.py`

### CALB

SNL 헐 감사 이후, **정규화 건강도**를 처음부터 잠근 새 아카이브다. `CALB.zip`(SHA-256 `540cabf7…1b5747`). 27셀 중 80% 교차 적격 1개(`CALB_0_B188`). inconclusive. Tongji만 표본 부족 백업으로 열어 두었다.

- 결과: `results/calb_ppx_v1/results.json`
- 코드: `experiments/calb_ppx_v1.py`

### Tongji

`Tongji.zip`(SHA-256 `d2ea02ba…aa42fb`). 28셀 중 적격 17. 학습 최소 정규화 건강도가 0.8000이라, 80% 교차 **이전** 검증·테스트 행이 그 아래로 내려갈 수 없다. infeasible. 이 구조 때문에 다음 고호트부터 학습 경계를 **교차 전 학습 건강도의 25백분위수**로 잠갔다. Tongji에 그 컷을 소급 적용하지 않는다. PAE가 같은 자료를 쓴 적이 있으므로, 나중에 점수가 나와도 dataset-level untouched가 아니라 PP model-level이다.

- 결과: `results/tongji_ppx_v1/results.json`
- 코드: `experiments/tongji_ppx_v1.py`

## Stanford — 유일한 채점

`Stanford.zip`(SHA-256 `55c8766e…5b4cf7`), Cui et al. 2024 파우치 41셀. BatteryLife 설명상 충방전 프로토콜은 같고 **포메이션만 다르다.** 프로토콜은 다운로드 전에 썼고, 25백분위수 학습 경계를 처음부터 고정했다.

| 항목 | 값 |
|---|---|
| 적격 / 학습 / 검증 / 테스트 셀 | 41 / 24 / 8 / 9 |
| 학습 건강도 25백분위수 | 0.9242 |
| 실제 학습 최소 건강도 | 0.9242 |
| 학습 / 검증 / 테스트 행 | 11407 / 1572 / 2427 |
| 검증·테스트 헐 밖 비율 | 100% |
| 선택 executor | unbounded (constrained executor 미승인) |
| 선택 폭·학습률 | 32, 1e-3 |

검증 Regular(216–224)에서 이미 validation MSE가 PP 2956, MLP 2127이었다. 테스트 Ref를 보기 전에 prior가 도움이 되지 않았다.

| 모델 | pooled R² | unit-macro R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| PP-X v1 ensemble | **−0.228** | −0.038 | 93.65 | 72.50 |
| matched MLP ensemble | **0.034** | 0.209 | **83.05** | **60.85** |

셀 부트스트랩 `MLP RMSE − PP RMSE` 평균 −10.57, 95% CI **[−13.58, −6.65]**, P(양수)=0.0001. `confirmatory_success = false`.

사전순 분할이 Regular 191–230 뒤에 Ref_100/101/102를 두어, 테스트 9셀 중 3개가 기준 포메이션이다(2427행 중 893행). 사후 분석용 Regular/Ref 분해는 확증 숫자가 아니다.

| 테스트 구간 | 행 | 실제 RUL 평균 | PP 평균 | MLP 평균 | pooled R² PP / MLP |
|---|---:|---:|---:|---:|---|
| Regular 225–230 | 1534 | 129 | 89 | 114 | 0.600 / 0.742 |
| Ref 100–102 | 893 | 156 | 31 | 43 | −1.174 / −0.771 |
| 전체 | 2427 | 139 | 68 | 88 | −0.228 / 0.034 |

같은 건강도(~0.87)에서 Ref는 남은 수명이 더 긴데 PP는 약 30사이클로 깎는다. Regular만 봐도 MLP가 이긴다. 헐 구간이 0.80–0.924라 주표의 깊은 용량 꼬리가 아니라 중기 RUL이다. Ref를 빼고 다시 나누지 않는다.

- 결과: `results/stanford_ppx_v1/results.json`
- 예측: `results/stanford_ppx_v1/predictions.npz`
- 코드: `experiments/stanford_ppx_v1.py`
- 프로토콜: `protocols/STANFORD_PPX_V1_PROTOCOL.md`

## 이 큐가 말하는 것

1. 1차 주표 성공은 같은 열화 법칙과 말기 용량 외삽에서의 개발 증거로 남는다. 새 고호트 일반화로 재분류하지 않는다.
2. 게이트에서 멈춘 다섯 건은 PP 성능 실패가 아니라 **평가 계약 실패**다. 억지 채점은 확증을 강하게 만들지 않는다.
3. Stanford는 계약이 열린 뒤의 음수 확증이다. prior가 포메이션·중기 수명 척도에 맞지 않으면 PP-X v1은 MLP만 못하다.
4. 표본 부족 백업으로만 열어 둔 `MICH_EXP`는 Stanford가 점수를 냈으므로 이 큐에서 열지 않는다.

다음 확증은 실패한 아카이브의 컷을 고치는 것이 아니라, 계약을 유지한 새 미개봉 고호트이거나, prior가 틀릴 때 거절하는 게이트의 독립 적용이다.
