# XJTU-SY untouched distance-shell 결과

## 증거 순서

- shell gate 구현 동결: `f6eb4c3`
- XJTU split·feature·metric protocol 공개 동결: `a3e8999`
- 이후 원자료를 처음 다운로드하고 feature를 추출했다.
- `gate_decision_pretest.json`을 먼저 저장한 후 test RUL을 계산했다.

데이터는 XJTU-SY의 15개 complete run-to-failure bearing이다. 조건2(2250 rpm,
11 kN) 5개를 train, 조건1(2100 rpm, 12 kN) 5개를 validation, 조건3(2400 rpm,
10 kN) 5개를 untouched test로 사용했다. 원자료 SHA-256은
`3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10`이다.

## Test label을 열기 전 gate 결정

validation과 source의 정규화 운전조건 거리는 모두 `sqrt(2)=1.414`이며 고정 shell
`[1,2)`에 들어간다.

| validation shell 증거 | 값 | 고정 기준 | 통과 |
|---|---:|---:|---|
| 표본 수 | 581 | ≥20 | 예 |
| Ridge 대비 PP MSE gain | 47.17% | ≥0% | 예 |
| 정규화 seed disagreement | 0.982 | ≤0.25 | **아니오** |

따라서 test label 없이 `ABSTAIN`했고 prediction coverage는 `0/6,999 = 0%`였다.

## 봉인 해제 후 accuracy audit

Gate가 거절한 뒤에만 test RUL을 만들고 성능을 계산했다. 이 수치는 실제 서비스
prediction이 아니라 abstention이 옳았는지 확인하는 audit다.

| 모델 | pooled R² | pooled RMSE | unit-macro R² |
|---|---:|---:|---:|
| Ridge/affine | -1.477 | 1,136.1 | -1.360 |
| 일반 NN ensemble | -1.565 | 1,156.0 | -2.973 |
| PP ensemble | **-1.308** | **1,096.7** | -2.399 |
| support-PP ensemble | -1.317 | 1,098.7 | -2.320 |

PP는 실패한 모델 중 상대적으로 가장 낮은 pooled error였지만 R²가 음수이므로 예측
성공으로 세지 않는다. 다섯 PP seed의 pooled R²는 `-1.378, -1.357, -0.903,
-1.477, -1.477`이었다.

## 해석

이 실험은 untouched 데이터에서 PP 정확도가 일반화됐다는 증거가 아니다. 대신 고정된
distance-shell gate가 validation의 큰 seed 불일치를 이용해 실제 실패한 조건을 test label
없이 거절했다는 첫 prospective abstention 증거다. 단 한 데이터셋의 거절이므로 실패
검출률을 일반화할 수 없고, prediction coverage가 0%라는 한계도 그대로 남는다.

다음 독립 cohort에서는 gate가 일부 표본을 승인하면서 selective risk를 낮추는지 확인해야
한다. 논문 표에서는 accuracy coverage와 abstention coverage를 분리하고, 이 XJTU 결과를
`correct abstention`, `predictive success`에는 포함하지 않는다.
