# 동일 조건 PFN 비교

## C-MAPSS FD002+FD004 — 완료

- split: 기존 strict OP1–OP3 convex-hull 밖 final 2,616행
- 입력: 동일 robust 115 features
- PFN train cap: 3,000행
- seeds: 42–46
- 주 지표: raw pooled R²

| 모델 | pooled R² | seed별 값 |
|---|---:|---|
| PFN v3 | 0.607±0.029 | 0.634, 0.593, 0.634, 0.558, 0.616 |
| 기존 staged PP | **0.749±0.015** | 기존 동결 결과 |
| exact-hull support-adaptive PP | 0.747±0.011 | 0.736, 0.734, 0.761, 0.750, 0.757 |

Support-adaptive PP와 PFN 차이는 **+0.140**이며 PP가 PFN의 5개 seed를 모두
상회했다. 다만 새 gate는 validation에서 `beta=0`이 선택되어 기존 PP보다
향상되지 않았다. 따라서 C-MAPSS에서는 support 감쇠가 아니라 soft anchor만
작동했고, 기존 staged PP와 사실상 동률이다.

PFN seed 변화에는 모델 random state와 3,000행 random subsample 변화가 함께
포함된다. 이는 기존 PFN 실행기의 seed 정의를 그대로 따른 것이다.

## HUST·Virkler·NASA — 다운로드된 로컬 v3 완료

로그인이 필요 없는 로컬 `tabpfn-v3-regressor-v3_default.ckpt`를 사용했다. 입력,
frozen split, test 행, 최대 3,000 equal-unit train 행과 seeds 42–46을 맞췄다.
HUST는 PP도 동일 3,000행으로 다시 학습했다. Virkler와 NASA는 원래 학습행이
cap 이하여서 full과 same-budget이 같다.

| 데이터 | 로컬 PFN v3 | Support-adaptive PP same-budget | PP 이득 |
|---|---:|---:|---:|
| HUST | 0.218±0.030 | **0.820±0.055** | +0.602 |
| Virkler | 0.621±0.066 | **0.886±0.025** | +0.265 |
| NASA health | -0.691±0.009 | **0.513±0.002** | +1.204 |
| C-MAPSS FD002+004 | 0.607±0.029 | **0.747±0.011** | +0.140 |

PFN seed별 값:

- HUST: 0.224, 0.188, 0.257, 0.181, 0.241
- Virkler: 0.627, 0.748, 0.576, 0.586, 0.569
- NASA: -0.698, -0.706, -0.684, -0.684, -0.684
- C-MAPSS: 0.634, 0.593, 0.634, 0.558, 0.616

네 데이터 모두 평균 기준 PP 우위다. C-MAPSS는 PP가 PFN 5/5 seed를
상회했다. 다른 세 데이터의 표는 모델별 5-seed 분포 비교이며 seed를 서로
일대일 대응시킨 통계 검정은 아직 수행하지 않았다. 전 과정은 이미 관찰한
데이터에 대한 retrospective 비교다.
