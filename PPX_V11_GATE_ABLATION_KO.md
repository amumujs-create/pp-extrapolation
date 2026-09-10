# PP-X v1.1 safety-gate ablation

기록: 박진서, 2026-09-10.

## 설계

Stanford의 기존 split, strict-tail 경계, 입력, 네트워크 grid와 seeds 42–46을
유지하고 v1.1에서 다음 세 arm을 비교했다.

1. `prior always-on`: validation RMSE가 가장 낮은 positive-trust 후보를
   승인문 없이 사용
2. `v1.1 validation gate`: 2% 상대 RMSE 개선과 physical-unit bootstrap
   95% 하한 양수를 요구
3. `matched MLP`: trust=0 하위모형

Stanford 결과를 본 뒤 수행한 **post-test development ablation**이며 새 확증이
아니다. 기존 v1.0 구성요소 ablation과 결과를 삭제하거나 대체하지 않는다.

## 결과

| Arm | Test pooled R² | RMSE | MAE |
|---|---:|---:|---:|
| Prior always-on, trust 0.02 | **0.105** | **79.96** | **59.39** |
| v1.1 validation gate | 0.087 | 80.77 | 59.98 |
| Matched MLP | 0.087 | 80.77 | 59.98 |

Positive-trust 후보의 validation 상대 RMSE 이득은 0.13%로 2% 문턱보다
낮아 거절됐다. Gate와 matched MLP의 최대 예측 차이는 0.0이다.

## 해석

이 데이터에서는 always-on 후보가 사후 test 점수로는 gate보다 R² 0.018
높았다. 따라서 “gate가 항상 정확도를 올린다”는 주장은 틀리다. 방어 가능한
주장은 다음이다.

> v1.1 gate는 test 최고점을 고르는 장치가 아니라, test를 보기 전에
> validation 근거가 부족한 prior를 포기하고 exact matched MLP로 후퇴시키는
> 안전 장치다.

재현:

- `experiments/stanford_ppx_v11_gate_ablation.py`
- `results/stanford_ppx_v11_gate_ablation/results.json`
- `results/stanford_ppx_v11_gate_ablation/predictions.npz`
