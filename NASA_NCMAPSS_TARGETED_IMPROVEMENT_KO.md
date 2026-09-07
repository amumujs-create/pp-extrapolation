# NASA·N-CMAPSS 표적 PP 개선 결과

## 채택 결과

| 데이터 | 기존 PP pooled R² | 최종 PP pooled R² | Engression | macro R² | 판정 |
|---|---:|---:|---:|---:|---|
| NASA battery | 0.571 | **0.584** | 0.549 | 0.616→**0.661** | causal multiscale PP 채택 |
| N-CMAPSS | **0.937** | **0.937** | 0.932 | 유지 | context head 5/5 seed 거부, 기존 PP 보존 |

NASA 개선량은 pooled R² +0.0125이고 unit-macro R²는 +0.0453이다. seed별 pooled R²는 0.580, 0.584, 0.592, 0.583, 0.580으로 안정적이었다.

## NASA 구조

기존 regime-spline PP는 현재 scalar health 하나만 입력받았다. 개선 모델은 미래값이나 최종수명을 입력하지 않고, 각 시점까지 관측된 health history에서 3·5·10 시점 평균, 감소율, 변동성과 변화율 가속도를 계산한다. latent-regime PP가 이 causal multiscale context로 local residual과 regime transition을 학습한다. `short`, `multiscale`, `moments` preset과 separation regularizer는 각 outer fold의 validation battery에서만 선택했다.

| 배터리 | 기존 PP RMSE | causal-history PP RMSE | Engression RMSE |
|---|---:|---:|---:|
| B0005 | **6.033** | 7.385 | 5.049 |
| B0006 | 12.031 | **7.729** | 9.623 |
| B0007 | **21.151** | 21.784 | 21.456 |
| B0018 | 8.731 | **8.633** | 13.629 |

B0006의 큰 개선과 B0018의 보존이 pooled·macro 점수를 높였다. B0005와 B0007은 악화했으므로 모든 배터리에 균일한 개선은 아니다.

## 폐기한 구조

1. **사후 context ridge head**: NASA 0.571→0.563. B0018은 좋아졌지만 B0006이 악화했다. N-CMAPSS에서는 5개 seed 모두 validation이 identity를 선택해 0.937을 정확히 보존했다.
2. **health-only affine backbone + history residual**: NASA 0.563. context를 affine path에서 완전히 제거하면 부족했다.
3. **scalar/history dual-view PP**: validation-selected mixture가 NASA 0.547. 한 validation battery의 view 비율이 다른 test battery로 전이되지 않았다.

따라서 모델 평균이나 test-selected 혼합으로 얻은 최고점은 채택하지 않았다. 최종 NASA 값 0.584는 하나의 causal-history latent PP 결과다.

## 해석

NASA에서는 단일 health 좌표보다 causal rate history가 유효했지만, validation battery 하나만으로 배터리별 최적 view를 판별하는 것은 불가능했다. 다음 개선은 output mixture가 아니라 train 내부 여러 battery의 pseudo-tail에서 regime encoder를 학습하는 nested unit-disjoint meta-training이어야 한다.

N-CMAPSS는 현재 개선 여지가 작다. 충분한 관측이 있는 두 test engine에서 이미 Engression보다 낮은 RMSE이고, 추가 context 보정은 validation 근거가 없어 자동 거부됐다. 이 데이터에서는 점수를 더 맞추기보다 새로운 test engine을 늘려 통계력을 확보하는 것이 우선이다.

## 재현

- 채택 NASA: `experiments/nasa_causal_multiscale_pp.py`, `results/nasa_causal_multiscale_pp_v1/`
- 거부된 context head: `experiments/context_conditioned_pp_nasa_ncmapss.py`
- 거부된 dual view: `experiments/nasa_dual_view_pp.py`
- health-only backbone 결과: `results/nasa_causal_multiscale_pp_v2/`
