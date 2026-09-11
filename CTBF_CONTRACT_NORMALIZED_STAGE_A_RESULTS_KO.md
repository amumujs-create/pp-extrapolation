# Contract-normalized CTBF Stage A 결과

작성자: 박진서  
상태: retrospective development; multi-timescale 주가설 실패,
contract-normalized direct-velocity CTBF 확인 필요

## 사전 판정

사전 동결한 Stage A 통과 규칙은 multi-timescale weak-rate CTBF를 주효과로
정했다. Stanford에서 이 arm의 RMSE가 78.586으로 기준 75를 넘었고 lag-1
CTBF보다 10.8% 나빴으므로 **Stage A는 실패**다. 이 결과를 바꾸거나
path-consistency Stage B로 자동 진행하지 않는다.

## 중요한 탐색적 결과

Prior를 사용하지 않은 contract-normalized direct-velocity CTBF는 두
데이터셋 모두에서 가장 좋은 pooled RMSE를 기록했다.

| Cohort | direct MLP | lag-1 CTBF | multi-rate CTBF | direct-velocity CTBF | stored PP-X v1.1 |
|---|---:|---:|---:|---:|---:|
| Stanford | 90.407 | 70.930 | 78.586 | **71.382** | 83.055 |
| ISU 250 mAh | 1.563 | 1.542 | 1.383 | **1.233** | 1.717 |

ISU의 cell-specific normalized failure boundary는 0.701–0.873이었고 중앙값은
0.731이었다. 이전 실험의 고정 0.8은 실제 contract와 크게 달랐다.
Contract normalization 후 direct-velocity CTBF의 ISU RMSE는 2.745에서
1.233으로 55.1% 감소했다.

## 사후 통계 진단

이 비교는 주가설 실패 후 수행한 탐색적 분석이므로 confirmatory evidence로
사용하지 않는다.

- Stanford direct-velocity CTBF vs direct MLP
  - unit bootstrap RMSE 개선: 18.43 cycles
  - 95% CI: [7.98, 25.94]
  - 개선확률: 1.000
  - unit wins: 8/9
- ISU direct-velocity CTBF vs direct MLP
  - unit bootstrap RMSE 개선: 0.373 cycles
  - 95% CI: [-0.007, 0.832]
  - 개선확률: 0.9718
  - unit wins: 37/46

## Multi-timescale prior 판정

- Stanford: multi-rate가 lag-1보다 7.84 cycles 악화
  - 95% CI: [-9.96, -5.90]
  - unit wins: 0/9
- ISU: multi-rate가 lag-1보다 0.164 cycles 개선
  - 95% CI: [0.072, 0.310]
  - unit wins: 29/46

따라서 단순 median velocity prior는 범용 개선이 아니다. 데이터셋마다
관측주기와 rate noise가 달라 같은 집계 규칙을 강제할 수 없다.

## 해석

현재 가장 유망한 모델적 발견은 weak-rate prior가 아니라 다음 구조다.

\[
z_{it}
=
\frac{q_{it}-q_{f,i}}{q_{0,i}-q_{f,i}},
\qquad
\widehat T
=
\int_0^{z_{it}}\frac{du}{v_\theta(u,c_{it})},
\qquad
v_\theta>0.
\]

즉 unit-specific failure contract로 상태좌표를 정규화하고, positive velocity
field의 first-passage time으로 RUL을 정의하는 것이 핵심이다.

## 다음 단계

1. multi-timescale Stage B는 사전 규칙에 따라 중단한다.
2. direct-velocity 구조를 새로운 가설로 별도 동결한다.
3. UConn 1.2 Ah 등 열지 않은 contract-compatible cohort에서 확인한다.
4. 확인 전에는 PP-X 메인 모델을 교체하거나 성능 우위를 주장하지 않는다.

## 재현

```bash
PYTHONPATH=src /opt/anaconda3/bin/python \
  experiments/ctbf_contract_normalized_stage_a.py
```

원시 결과:

- `results/ctbf_contract_normalized_stage_a/results.json`
- `results/ctbf_contract_normalized_stage_a/predictions.npz`
