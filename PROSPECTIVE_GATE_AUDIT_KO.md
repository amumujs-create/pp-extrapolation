# 사전 PP 게이트 prospective audit

## 결론

XJTU와 Oxford 이후 남아 있던 “untouched 외부 데이터에서 사전 게이트가 성공했다”는
공백을 해소하기 위해 순차적으로 프로토콜을 커밋한 뒤 공개 Na-ion 코호트를 한 번씩
평가했다. 결과는 **현재 정적 게이트의 완성 주장을 지지하지 않는다.** 거리,
validation 상대 성능, seed disagreement만으로는 새로운 cell의 미래 health–RUL 관계
변화를 안정적으로 판별하지 못했다.

반면 다음 두 사실은 새로 확인됐다.

1. 사전 거절은 실제 실패 route를 걸러냈다. Group 4의 평가 가능한 cell에서 PP R²
   `0.847`, MLP R² `0.959`였고 고정 게이트는 MLP를 선택했다.
2. 사전 승인의 성공 사례도 존재한다. 공식 80%-EOL test cohort의
   `270040-6-6-26`에서 PP R² `0.563`, MLP R² `-0.196`이었다. 그러나 게이트가 다른
   네 cell도 승인했으므로 선택기의 전체 성공으로 주장할 수 없다.

## 순차 실험과 변경 이력

각 단계의 코드와 규칙은 해당 cohort를 받기 전에 별도 commit으로 고정했다.

| 단계 | 사전 고정 commit | 새 cohort | 결정 | outcome | 해석 |
|---|---|---|---|---|---|
| v1 | `1e3e638` | Na-ion group 1 | 거절 | PP 0.578, MLP -0.679 | absolute validation R² 조건의 false reject |
| v2 | `7324703` | Na-ion group 2 | 승인 | PP -0.544, MLP 0.455 | 소수 PP tail 폭주로 false accept |
| v3 | `ac46281` | Na-ion group 3 | 승인 | projected PP 0.653, MLP 0.748 | 생존 투영은 폭주를 줄였으나 unit 혼합 실패 |
| v4 | `e9d841c` | Na-ion group 4 | 전부 거절 | 평가 cell PP 0.847, MLP 0.959 | 올바른 거절, PP coverage 0% |
| v4 복제 | `773850a` | Na-ion group 5 | 전부 승인 | PP -0.139, MLP 0.114 | 1/2 unit만 올바른 승인 |

v1--v4 복제는 파일의 마지막 cycle을 RUL=0으로 둔 탐색이었다. 이후 공식 BatteryLife
label extractor를 감사해 이 정의가 잘못됐음을 확인했다. 공식 규칙은 nominal capacity
1.0 Ah의 80% 도달 cycle을 EOL로 사용하고, 마지막 SOH가 0.825 이상이면 cell을
제외한다. 실제 group 5의 기록 종료 SOH는 `0.15--0.91`이어서 recording termination은
공통 failure가 아니었다. 따라서 위 다섯 단계는 **protocol failure와 gate 개발
이력**이며 PP 성능 표의 확증 결과로 사용하지 않는다.

## 공식 80%-EOL prospective test

수정 프로토콜은 commit `691ecb2`에서 test 파일 다운로드 전에 고정했다.

- train: groups 1--3의 80%-EOL 적격 cell 19개
- validation: group 4의 적격 cell 5개
- untouched test: group 6/7/8에서 공식 목록으로 고정한 5개 cell
- 평가 행: 538개 causal late-tail point
- validation: projected PP R² `0.645`, MLP R² `0.366`; PP MSE gain 44.1%; normalized
  seed disagreement 0.025
- 사전 결정: 5/5 cell PP 승인

| test cell | PP R² | MLP R² | 실제 우세 | 사전 결정 |
|---|---:|---:|---|---|
| 270040-6-2-30 | -2.096 | 0.526 | MLP | PP |
| 270040-6-6-26 | **0.563** | -0.196 | PP | PP |
| 270040-6-8-24 | 0.778 | **0.813** | MLP | PP |
| 270040-7-1-23 | 0.687 | **0.819** | MLP | PP |
| 270040-8-5-16 | -16.345 | **-4.503** | MLP | PP |
| **pooled** | 0.457 | **0.470** | MLP | PP |

Unit route accuracy는 `1/5=20%`였고 prospective gate success criterion은 실패했다.
Pooled 격차는 작지만, cell별 실패가 크므로 평균 수치만으로 성공이라 부르지 않는다.

## 왜 정적 게이트가 실패하는가

현재 prefix descriptor는 boundary health, 전체/초기/최근 slope, 곡률, 잡음을 측정한다.
이 값은 “지금까지 비슷하게 열화했는가”를 확인하지만 다음 정보를 포함하지 않는다.

- 관측 경계 이후 발생하는 knee 또는 regime transition
- 동일 초기 궤적 뒤에 달라지는 열화 가속도
- protocol ID만으로 표현되지 않은 cell별 latent susceptibility

따라서 test prefix가 train/validation support 안에 있고 PP가 validation에서 안정적으로
이겨도 미래 법칙은 달라질 수 있다. XJTU의 opposite-ray처럼 관측 시점에 드러난 shift는
거절할 수 있지만, 아직 발생하지 않은 transition은 정적 covariate gate로 식별할 수
없다.

## 논문에서 가능한 주장

- XJTU: label-free geometry/disagreement gate의 올바른 실패 거절 사례.
- Na-ion group 4: 새로운 prospective 올바른 거절 사례.
- Na-ion 80%-EOL: 정적 사전 게이트의 false-accept 한계를 보여 주는 prospective
  negative validation.
- 생존 지지집합 투영: `0 <= RUL(t) <= L_known_max-age(t)`로 tail 폭주를 줄이는
  model-side 안전 모듈. Group 2 개발 결과에서 PP R²를 -0.544에서 0.805로 복구했지만
  독립적인 보편 성능 주장은 하지 않는다.

“사전 게이트가 untouched 외부 데이터에서 성공했다”는 문장은 아직 사용하지 않는다.
대신 **정적 applicability certificate의 식별 한계를 prospective로 규명했다**고 쓰고,
후속 모델은 예측 중 새 관측이 들어올 때 change point와 posterior predictive surprise를
누적하는 동적 regime gate로 설계한다. 이 후속 모델은 별도 untouched cohort에서 다시
검증해야 한다.

## 재현 파일

- 공식 EOL 프로토콜: `protocols/NAION_80EOL_PROSPECTIVE_GATE_PROTOCOL.md`
- 실행 코드: `experiments/naion_80eol_prospective_gate.py`
- 사전 결정: `results/naion_80eol_prospective_gate/gate_decision_preoutcome.json`
- 전체 결과: `results/naion_80eol_prospective_gate/results.json`
- 초기 순차 audit 프로토콜과 결과는 `protocols/NAION_*` 및 `results/naion_*`에 보존한다.

