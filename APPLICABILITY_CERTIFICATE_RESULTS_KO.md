# PP Applicability Certificate 개발 감사

Test label 없이 PP가 사용할 만한 데이터인지 판단하기 위해 다음 validation 조건을 고정해 감사했다.

- PP의 affine 대비 mean-seed validation MSE 개선 ≥ 1%
- NN residual activity ≥ validation target SD의 0.5%
- normalized seed disagreement ≤ 0.5
- 하나 이상의 causal feature와 RUL의 absolute Spearman ≥ 0.2

## 결과

| 데이터 | Certificate | 실제 test 성공 | validation gain | residual activity | test PP ensemble R² |
|---|---|---|---:|---:|---:|
| HUST | 승인 | 성공 | 49.8% | 50.2% | 0.773 |
| Virkler | 승인 | 성공 | 84.1% | 89.7% | 0.888 |
| RWTH | 승인 | 성공 | 28.9% | 24.7% | 0.507 |
| Sunwoda | 거절 | 성공 | 0.6% | 1.4% | 0.865 |
| MICH | 거절 | 실패 | ≈0% | ≈0% | -1.522 |
| FEMTO | 거절 | 실패 | ≈0% | 0% | -1.378 |

6개 중 5개를 맞췄고 false accept는 0, false reject는 1이었다. 이 certificate는 성능이 가능한 데이터를 모두 포괄하기보다 실패 모델을 안전하게 거절하는 쪽이다.

## 실패 원인

MICH와 FEMTO 모두 validation 내부 feature–RUL 상관은 높았다. 그러나 그 상관만으로 affine 경로보다 나은 transferable residual을 학습하지 못했다. 두 데이터에서 모든 PP seed가 epoch 0/동일 affine 결과로 수렴했고 residual activity가 사실상 0이었다.

- MICH: train과 source cohort 사이 health–RUL mapping 변화가 커서 train residual이 전이되지 않았다.
- FEMTO: 6개 Learning bearing, 한 개 validation bearing, 약한 진동 health marker 조합으로 cross-bearing residual을 확인할 수 없었다.
- Sunwoda: PP 자체는 성공했지만 affine 대비 validation 추가 이득이 1% 미만이라 certificate가 보수적으로 거절했다.

따라서 실패의 핵심은 convex-hull 거리 하나가 아니라 `validation에서 재현되는 비선형 residual evidence`의 부재다. 이 결과는 retrospective development audit이며 독립 confirmatory 성능으로 주장하지 않는다.
