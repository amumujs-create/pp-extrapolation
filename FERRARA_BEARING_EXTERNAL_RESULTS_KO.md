# Ferrara 베어링 외부 고호트 결과 — 2026-09-09

## 결론

사전 고정한 E1--E3 train / E4 validation / E5--E6 test를 한 번 평가한 결과,
**Ferrara는 PP 성공 고호트가 아니다.** PP와 모든 학습 모델의 pooled R²가 음수였고,
test의 scalar degradation coordinate 밖 비율도 24.6%라 사전 정의한 strict
state-range 외삽 기준을 만족하지 않았다. 이 one-shot 결과는 이후 개발 결과로
덮어쓰지 않는다.

## 고정 모델 결과

| 모델 | pooled R² | RMSE (min) | MAE (min) |
|---|---:|---:|---:|
| Boundary-rate 진단값 | **-0.227** | **849.9** | **616.9** |
| Matched plain MLP, 5-seed ensemble | -5.871 | 2010.9 | 1866.3 |
| Elapsed-time 진단값 | -33.081 | 4478.6 | 3680.0 |
| FT-Transformer, validation-tuned 5-seed ensemble | -151.444 | 9472.0 | 8716.0 |
| Ridge, validation-selected alpha=1000 | -1090.363 | 25343.9 | 23339.8 |
| **Boundary-quotient PP, 5-seed ensemble** | **-2359.980** | **37276.5** | **34057.9** |
| PP frozen boundary-affine path | -2360.690 | 37282.1 | 34063.1 |

PP의 사전 성공 조건 `pooled R² > 0` 및 `RMSE < matched MLP`를 모두 실패했다.

## 단위별 PP 결과

| test bearing | 실제 tail 행 | PP R² | MLP R² |
|---|---:|---:|---:|
| E5 | 516 | -3074.148 | -7.624 |
| E6 | 147 | -6206.775 | -31.133 |

## 데이터 무결성과 수명

출판사가 공개한 SHA-256과 여섯 아카이브가 모두 일치했다. 추출된 trajectory
fingerprint 중복도 없었고 E5/E6은 각각 최소 tail 30개 기준을 통과했다.

| unit | load | event index | event time (min) | event provenance |
|---|---:|---:|---:|---|
| E1 | 4.0 kN | 4915 | 24575 | waveform 내 최초 20 g crossing |
| E2 | 4.0 kN | 1981 | 9905 | waveform 내 최초 20 g crossing |
| E3 | 4.0 kN | 2380 | 11900 | waveform 내 최초 20 g crossing |
| E4 | 3.0 kN | 668 | 3340 | 출판사가 확인한 terminal 20 g stop |
| E5 | 4.7 kN | 1717 | 8585 | waveform 내 최초 20 g crossing |
| E6 | 5.0 kN | 488 | 2440 | waveform 내 최초 20 g crossing |

## 왜 실패했는가

첫째, **고정 split에서 load와 lifetime 관계가 식별되지 않는다.** 4 kN train 세
베어링의 수명부터 9,905--24,575분으로 2.5배 다르다. 유일한 validation load인
E4 3 kN은 오히려 3,340분으로 가장 짧다. 따라서 E1--E4 full-development refit은
낮은 load가 짧은 수명이라는 표본 상관을 학습하고, 처음 보는 더 높은 load
4.7--5 kN에 반대 방향으로 크게 외삽했다. 물리적 load-life 효과와 unit 이질성을
네 unit만으로 분리할 수 없다.

둘째, **20 g margin은 현재 상태는 나타내지만 lifetime scale은 식별하지 못한다.**
실제 test tail RUL은 0--2,575분인데 PP 중앙 예측은 45,242분이었다. PP와 frozen
affine 결과가 거의 같으므로 NN seed 분산이 아니라 affine quotient scale bias다.
이는 앞선 Zn-ion 장수명 실패와 같은 종류의 구조적 실패다.

셋째, **이 split은 엄격한 health-range 외삽이 아니었다.** Full-development prefix의
running-peak 범위는 2.79--13.28 g이고, test tail의 24.6%만 이 범위를 벗어났다.
사전 기준 50% 미만이므로 unseen-unit/load late-tail 평가는 맞지만 strict scalar
state-range extrapolation 성공/실패 표에는 포함하지 않는다.

넷째, plain MLP도 seed 5개 모두 음수였고 ensemble R²는 -5.871이다. 가장 단순한
boundary-rate도 -0.227이므로 현재 feature와 split에서 일반 NN이 잘 풀고 PP만
실패한 경우가 아니다. 다만 PP의 경계 quotient가 잘못된 scale을 훨씬 강하게
증폭해 실패 폭이 가장 컸다.

## 논문에서의 처리

- 성공 외부 고호트로 세지 않는다.
- 사전 고정된 실패 사례와 PP applicability limitation으로 보존한다.
- `known boundary`만으로 충분하다는 주장을 폐기하고, boundary prior와 별도로
  **lifetime-scale identifiability**가 필요하다는 근거로 사용한다.
- 이후 Ferrara에서 load feature 제거, quotient clipping, alternate split 또는 새
  architecture를 시험하면 모두 post-hoc development로 표기한다.
- 다음 독립 비배터리 확증은 반복 unit이 더 많고 condition별 replicate가 있는
  데이터에서 해야 한다. Paderborn 17-bearing은 표본 수는 낫지만 feature와 모델을
  Ferrara 결과로 재설계한 뒤 그대로 동결해야 확증 의미가 생긴다.

재현 결과: `results/ferrara_bearing_external_locked_v1/results.json`  
예측: `results/ferrara_bearing_external_locked_v1/predictions.npz`  
코드: `experiments/extract_ferrara_bearing.py`,
`experiments/ferrara_bearing_external_eval.py`

