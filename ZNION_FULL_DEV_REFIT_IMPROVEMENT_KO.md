# Zn-ion PP full-development refit 개선

## 결과

기존 PP는 train prefix로 BQ를 학습하고 validation tail로 epoch을 선택했지만, 선택 후 validation prefix를 포함한 최종 재학습을 하지 않았다. 동일 config와 선택 epoch을 유지하면서 train+validation prefix 전체로 BQ를 다시 학습하고, 기존 RBF lifetime memory와 고정 gate를 그대로 사용했다.

| 모델 | pooled R² | unit-macro R² | RMSE |
|---|---:|---:|---:|
| 기존 alpha1000 RBF-regime PP | 0.822 | 0.375 | 약 97.7 |
| **full-development refit PP** | **0.909** | **0.690** | **69.612** |

| v3 unit | 기존 R² | refit R² |
|---|---:|---:|
| 209-1 | 0.772 | 0.798 |
| 412-1 | -0.405 | **0.397** |
| 446-1 | 0.757 | **0.877** |

Seeds 42–46의 개별 pooled R²는 0.909257535–0.909257578이었다. 현재 raw-cycle BQ residual의 보정 폭이 매우 작아 초기화 차이가 결과에 거의 영향을 주지 않은 것과 일관된다.

## 무엇이 개선됐는가

새로운 test label이나 test feature를 학습에 넣은 것이 아니다. validation으로 config와 epoch을 고른 다음, 최종 모델이 사용할 수 있는 development prefix supervision을 모두 포함해 동일 epoch만큼 재학습했다. RBF memory는 이전 모델과 동일하게 development cell의 prefix descriptor와 EOL label을 사용한다.

이 개선의 중심은 새로운 NN layer가 아니라 **누락됐던 final refit 계약**이다. 특히 validation의 장수명 cell이 RBF memory에만 들어가고 BQ affine fitting에는 빠져 있던 정보 비대칭을 줄였다.

## 함께 실행한 구조 후보

출력 정규화 additive BQ와 multiplicative BQ도 비교했다. v3 pooled R²는 0.777–0.803으로 기존 raw-cycle PP 0.822보다 낮아 채택하지 않았다. Development nested LOO에서도 단기 unit 일부는 좋아졌지만 장수명 unit을 통째로 제외한 fold에서 무너졌다.

따라서 현재 채택 후보는 `raw additive BQ + full-development fixed-epoch refit + 기존 RBF regime path`다. scale-aware residual은 장수명 학습 unit이 추가될 때 다시 평가한다.

## 증거 범위

v3 outcome을 이미 본 뒤 실행한 개발 replay이므로 0.909를 untouched confirmation이라고 부르지 않는다. 다음 독립 cohort에서 다음을 고정해야 한다.

1. alpha=1000, residual bound=0.5, raw additive mode.
2. train/validation에서 선택한 seed별 epoch.
3. config 선택 후 train+validation prefix full-development refit.
4. 기존 RBF temperature와 slope gate.
5. PP와 baseline 모두 같은 full-development refit 및 EOL supervision 계약.

재현 코드: `experiments/znion_scale_aware_rbf_replay.py`  
원시 결과: `results/znion_scale_aware_rbf_replay_v1/results.json`, `full_dev_refit_predictions.npz`
