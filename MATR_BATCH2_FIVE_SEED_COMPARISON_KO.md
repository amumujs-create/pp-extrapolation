# MATR batch 2 동일 seed 5회 비교

모든 모델은 seeds `42, 43, 44, 45, 46`으로 평가했다. PP·CPGRU·CPTransformer·V-REx는 최종 PP의 strict-tail test 733행을 사용한다. 표의 `mean ± SD`는 다섯 번의 개별 pooled R² 평균과 표본 표준편차이고, `prediction ensemble`은 다섯 예측을 행별 평균한 뒤 한 번 계산한 pooled R²다. 두 통계는 서로 바꾸어 쓰지 않는다.

| 모델 | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | mean ± SD | prediction ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|
| **최종 PP** | **0.912** | **0.888** | **0.772** | **0.802** | **0.888** | **0.852 ± 0.061** | **0.862** |
| V-REx | 0.757 | −0.224 | −0.237 | −0.354 | 0.291 | 0.046 ± 0.468 | 0.850 |
| TabPFN v3† | 0.580 | 0.668 | 0.559 | 0.627 | 0.647 | 0.616 ± 0.041 | 0.618 |
| BatteryLife CPGRU | 0.721 | 0.054 | 0.664 | 0.015 | 0.476 | 0.386 ± 0.334 | 0.537 |
| BatteryLife CPTransformer | 0.851 | −1.412 | −1.001 | 0.591 | 0.565 | −0.081 ± 1.044 | 0.380 |

PP는 다섯 seed 모두에서 각 경쟁모델보다 높은 pooled R²를 냈다. 같은 seed로 짝지은 one-sided Wilcoxon 검정은 각 비교에서 최소 가능한 정확 p-value `0.03125`다. paired t-test p-value는 V-REx `0.0133`, TabPFN `0.00085`, CPGRU `0.0354`, CPTransformer `0.1120`이다. 표본이 5개뿐이고 CPTransformer 분산이 매우 커서 t-test 하나만으로 우월성을 주장하지 않는다. 논문에는 effect size, seed 승패 `5/5`, 정확 Wilcoxon 결과를 같이 보고한다.

† TabPFN은 CPU 제약으로 train 1,000행을 equal-unit sampling한 보조 비교다. 완전한 동일 train-budget 비교로 주장하지 않는다.

CPGRU의 이전 `0.912`는 train 11,553행 protocol 결과다. 위 CPGRU는 최종 PP와 같은 train 11,552행으로 재실행한 값이다.

재현 코드: `experiments/matr_batch2_pp_five_seed_replay.py`, `experiments/batterylife_strict_tail_adapter.py`  
원시 결과: `results/matr_batch2_pp_five_seed_replay/`, `results/batterylife_strict_tail_matrb2_final_pp/`, `results/batterylife_strict_tail_matrb2_final_cptransformer/`
