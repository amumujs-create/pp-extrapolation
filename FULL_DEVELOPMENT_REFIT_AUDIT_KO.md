# PP full-development refit 감사

## 결과

Validation으로 config와 epoch을 고른 뒤 train+validation의 prefix를 합쳐 고정 epoch 재학습하는 동일 계약을 Zn-ion과 Na-ion에 적용했다. 비교 MLP도 같은 절차로 재학습했다.

| 데이터 | 기존/no-refit PP | refit PP | refit MLP | refit PP macro R² |
|---|---:|---:|---:|---:|
| Zn-ion v3 | 0.822 | **0.909** | 0.528 | **0.690** |
| Na-ion 80%-EOL | 0.781 | **0.824** | 0.304 | **0.806** |

Zn-ion 세 unit과 Na-ion 다섯 cell에서 refit PP의 unit R²가 모두 양수였다. Zn-ion PP seeds 42–46의 pooled R²는 0.909257535–0.909257578로 거의 같았다.

공동 dual-scale PP의 기존 full-development refit도 재실행했다.

| 데이터 | 재현 pooled R² |
|---|---:|
| Sunwoda | 0.934 |
| RWTH | 0.842 |
| MICH | 0.751 |

17,645개 seed×test 예측에서 `abs(prediction-affine) <= margin×6` 위반은 0건이었다.

## 해석

Zn-ion 개선은 validation의 장수명 cell 정보가 RBF memory에만 들어가고 BQ fitting에는 빠졌던 불균형을 줄인 영향이 크다. Na-ion에서도 개선됐으므로 refit 효과가 Zn-ion 한 cohort에만 국한되지는 않았다.

이 실험은 이미 본 test를 재생한 사후 개발 결과다. 모델 구조의 untouched 확증이나 모든 일반 NN보다 우월하다는 증거가 아니다. 비교 MLP는 동일 정보와 refit 계약을 받았지만 architecture search를 하지 않았다.

로컬의 `naion_external_replication`, `naion_external_v2`, `v3`, `v4`는 앞선 게이트 개발 스크립트에서 이미 사용됐다. 이를 새로운 untouched cohort로 재분류하지 않는다. 다음 확증에서는 새 physical unit을 받기 전에 refit 절차, EOL, 최소 tail, 중복 검사와 성공 기준을 고정해야 한다.

## 재현

- Zn-ion 및 동일 refit MLP: `experiments/znion_scale_aware_rbf_replay.py`
- Na-ion 및 동일 refit MLP: `experiments/naion_full_dev_refit_replay.py`
- 공동 battery PP 회귀 검사: `experiments/bq_dual_scale_final_replay.py`
- 결과: `results/znion_scale_aware_rbf_replay_v1/`, `results/naion_full_dev_refit_replay_v1/`, `results/bq_dual_scale_final_replay_v1/`
