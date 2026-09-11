# Tongji CY45-05 외삽 비교모델 결과

작성자: 박진서  
상태: retrospective (CCMR test 개봉 후)

## 요약

동일 feature·validation-only 선택으로 비교군을 돌렸을 때 **CCMR v2.0이
accuracy leader**였고, Engression보다 RMSE 개선과 max regret 모두
우세했다.

| 모델 | pooled R² | RMSE 개선 | max regret |
|---|---:|---:|---:|
| **CCMR v2.0** | **0.999976** | **18.00%** | **0%** |
| Engression | 0.999971 | 9.47% | 75.65% |
| Persistence | 0.999965 | 0% | 0% |
| Linear-mean GP | 0.998951 | -447% | 7417% |
| GroupDRO | 0.995677 | -1012% | 20688% |
| Monotone NN | 0.991069 | -1498% | 52750% |
| V-REx | 0.980560 | -2257% | 121819% |
| Linear-tail RBF | 0.936615 | -4157% | 430861% |
| TabPFN v3 | 0.536459 | -11412% | 3413310% |

## 계약

- 고호트: Tongji `CY45-05` (train/val/test 6254/1185/959 rows)
- CCMR: 봉인 `deployed` 예측 재사용 (재학습 없음)
- 비교군: V-REx, GroupDRO, Monotone NN, Linear-tail RBF, Engression,
  linear-mean GP, TabPFN v3
- 출력: nonnegative only (train 상한 미강제)
- seeds 42–46 ensemble

## 산출물

- `experiments/tongji_cy45_05_extrapolation_competitors.py`
- `results/tongji_cy45_05_extrapolation_competitors/results.json`
- `results/tongji_cy45_05_extrapolation_competitors/tongji_cy45_05_ensemble_predictions.npz`
