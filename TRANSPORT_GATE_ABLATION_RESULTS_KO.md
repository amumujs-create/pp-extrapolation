# PP error-transport gate 제거·correction-family 실험

## 질문

1. Dual gate가 PP만 승인하도록 만들어진 규칙인가?
2. 단순 bias correction으로도 같은 결과를 얻는가?
3. Seed와 validation-unit 증거가 bounded affine의 전이를 실제로 구별하는가?

동일한 MATR2019 train/validation/test split과 저장 checkpoint를 사용했다. 모든 보정 계수와
gate 결정은 validation에서 계산했다. 결과는 이미 확인한 split의 post-hoc method ablation이다.

## PP correction family

| correction | seed wins | validation-unit gain 95% CI | 무조건 적용 test R² | dual-gate final R² |
|---|---:|---:|---:|---:|
| identity | — | — | 0.257 | 0.257 |
| bias | 3/5 | [-478, 470] | 0.345 | 0.257 |
| group-balanced bias | 3/5 | [-556, 519] | 0.351 | 0.257 |
| **bounded affine** | **5/5, p=.03125** | **[157, 1443]** | **0.466** | **0.466** |

Offset만 고치는 두 방법도 test에서는 좋아졌지만 seed exact test와 validation-unit bootstrap을
통과하지 못했다. Bounded affine만 두 변동 축에서 재현됐다. 따라서 PP 개선은 아무 종류의
사후 보정을 붙인 결과가 아니라 validation에서 shape-preserving scale/offset transport가
반복적으로 확인된 경우다.

## FT에 동일 verifier 적용

| correction | seed wins | validation-unit gain 95% CI | 무조건 적용 test R² | dual-gate final R² |
|---|---:|---:|---:|---:|
| identity | — | — | 0.331 | 0.331 |
| bias | 0/5 | [-165, -30] | 0.335 | 0.331 |
| group-balanced bias | 0/5 | [-233, -16] | 0.319 | 0.331 |
| bounded affine | 2/5 | [-181, -10] | 0.376 | 0.331 |

FT의 bounded affine은 우연히 test R²를 0.376으로 높였지만 validation에서는 seed 2/5이고
unit-level mean gain의 CI도 완전히 음수였다. Dual gate는 이를 승인하지 않았다. 이는
false rejection일 수 있지만 test 결과를 보고 correction을 선택하지 않는 selective
protocol의 비용이다. 반대로 PP는 같은 verifier를 통과해 0.466으로 FT보다 높았다.

## Gate 제거 해석

- **No gate:** PP와 FT 모두 test에서 좋아지지만, test를 보기 전에는 어떤 보정을 운반할지
  구별할 근거가 없다.
- **Seed-only:** 초기화 반복성만 검사하고 특정 validation unit에 의존하는 오류를 놓칠 수 있다.
- **Unit-only:** 물리 unit 반복성은 보지만 학습 초기화에 따른 불안정성을 놓칠 수 있다.
- **Dual:** 두 조건을 모두 요구하여 coverage를 줄이는 대신 correction provenance를 명시한다.

이번 MATR에서는 PP affine이 두 단일 gate와 dual을 모두 통과하고 FT는 모두 실패하므로,
두 gate 중 하나를 제거했을 때의 실제 test 손상 차이는 나타나지 않는다. Dual의 추가가치를
정량화하려면 서로 다른 seed와 unit 판정이 엇갈리는 추가 데이터가 필요하다.

## 노벨티에 주는 근거

이 실험은 PP를 일반 calibration과 구분한다. 핵심은 affine map의 식이 아니라

`candidate correction → seed replication → physical-unit replication → transport or identity`

라는 실행 절차다. 같은 verifier가 PP와 FT를 다르게 판정하므로 PP 전용 hard-code가 아니다.
PAE가 관측 계약으로 학습 전 구조 prior를 컴파일한다면, PP는 학습 후 prediction error의
transportability를 검증한다. 두 논문의 역할 분리가 이 실험으로 더 명확해졌다.

기계 판독 결과: `results/transport_gate_ablation_v1/results.json`
