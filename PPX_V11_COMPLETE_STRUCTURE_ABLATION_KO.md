# PP-X v1.1 전체 구현 구조 ablation

기록: 박진서, 2026-09-10.

## 범위

기존 v1.0 ablation은 삭제하거나 재해석하지 않았다. 이번 분석은 구현된
v1.1 safety continuation의 모든 설정축을 대상으로 한다.

- 고호트: Stanford, ISU–ILCC 250 mAh
- trust: `0, .02, .05, .10, .20, .40`
- width: `16, 32`
- learning rate: `5e-4, 1e-3`
- weight decay: `2.0`
- seeds: `42–46`
- 2% validation margin on/off × unit-bootstrap on/off
- 총 **48 architecture×trust arm, 240 fits**

두 데이터 모두 이미 결과를 본 고호트이므로 retrospective mechanism
ablation이며 새 확증은 아니다. 계약은
`protocols/PPX_V11_COMPLETE_STRUCTURE_ABLATION_PROTOCOL.md`에 실행 전에
고정했다.

## Trust dose

각 trust에서 validation ensemble RMSE가 가장 낮은 architecture를 선택했다.

| Cohort | Trust | Val R² | Test R² | Test RMSE |
|---|---:|---:|---:|---:|
| Stanford | 0 | **0.615** | 0.034 | 83.05 cycles |
|  | .02 | 0.613 | 0.034 | 83.06 |
|  | .05 | 0.609 | 0.034 | 83.07 |
|  | .10 | 0.594 | 0.085 | 80.84 |
|  | .20 | 0.577 | **0.094** | **80.47** |
|  | .40 | 0.546 | 0.085 | 80.84 |
| ISU 250 mAh | 0 | 0.365 | 0.451 | 1.717 days |
|  | .02 | **0.386** | 0.555 | 1.546 |
|  | .05 | 0.384 | 0.557 | 1.543 |
|  | .10 | 0.379 | **0.557** | **1.542** |
|  | .20 | 0.372 | 0.534 | 1.582 |
|  | .40 | 0.363 | 0.522 | 1.601 |

Test 최고 trust는 Stanford .20, ISU .10이지만 validation이 선택한
always-on 후보는 두 곳 모두 .02다. Test에서 trust를 고르면 누수이므로
이 최고점은 선택 결과가 아니라 dose-response 설명값이다.

## Gate 2×2

| Cohort | Margin | Bootstrap | Prior 승인 | Test R² |
|---|---|---|---|---:|
| Stanford | off | off | yes | 0.034 |
|  | off | on | no | 0.034 |
|  | on | off | no | 0.034 |
|  | on | on | no | 0.034 |
| ISU 250 mAh | off | off | yes | 0.555 |
|  | off | on | no | 0.451 |
|  | on | off | no | 0.451 |
|  | on | on | no | 0.451 |

Stanford의 validation 상대 RMSE 이득은 **−0.28%**, bootstrap 95% CI는
`[−0.276, 0.208] cycles`, unit 승리는 4/8이다. ISU는 **+1.65%**,
bootstrap CI `[−0.145, 0.196] days`, unit 승리는 32/45다. ISU는 prior가
test에서 크게 좋아졌지만 2%와 bootstrap 문을 사전에 통과하지 못했다.

따라서 두 gate 요소는 이번 두 고호트에서 각각 단독으로도 prior를 거절한다.
이는 gate가 위험을 줄이는 대신 유효한 약한 prior까지 놓칠 수 있다는
보수성 비용을 직접 보여준다.

## Seed 안정성과 exact fallback

- Stanford fallback seed R²: `0.069, 0.046, −0.158, 0.055, 0.086`
- Stanford always-on .02: `0.069, 0.051, −0.162, 0.059, 0.078`
- ISU fallback: `0.454, 0.438, −0.218, 0.040, 0.266`
- ISU always-on .02: `0.439, 0.286, −0.155, 0.203, 0.270`
- Full gate가 거절했을 때 gated ensemble과 fallback의 최대 차이: 두 곳
  모두 **0.0**

Seed 하나는 두 고호트 모두 음수여서 개별 초기화 안정성은 충분하지 않다.
양의 ensemble R²만으로 seed-robust하다고 주장하지 않는다.

## 최종 해석

1. 작은 positive trust가 validation에서는 가장 자주 선택되지만 test 최적
   trust는 다르다.
2. 2% margin과 bootstrap은 중복적으로 보수적이며, 이번 두 곳 모두 prior를
   거절한다.
3. ISU에서는 그 거절이 실제 정확도 비용을 만든다. 따라서 v1.1은 PP
   성공률을 최대화하는 모델이 아니라 false approval을 줄이는 safety model이다.
4. trust=0 exact fallback은 구현상 검증됐다.
5. 두 고호트만으로 2%와 95% 문턱의 보편적 최적성을 주장할 수 없다.

재현:

- `experiments/ppx_v11_complete_structure_ablation.py`
- `results/ppx_v11_complete_structure_ablation/results.json`
- `results/ppx_v11_complete_structure_ablation/predictions.npz`
