# Stability-First RBPR v1.3 개발 결과

작성자: 박진서  
상태: Stanford·ISU를 이미 본 뒤 수행한 retrospective 강건성 개발

## 변경 목적

최고 R²나 특정 고호트의 단일 개선보다 physical-unit, fold, trust 변화에 대한
안정성을 우선한다. prior는 다음 조건을 모두 통과할 때만 사용한다.

- architecture × seed no-prior baseline portfolio
- leave-one-physical-unit-out fold에서 prior 채택률 80% 이상
- 동일 trust 합의율 60% 이상
- fold alpha IQR .25 이하
- OOF raw unit regret:
  - 평균 2% 이하
  - worst-20% CVaR 5% 이하
  - 최대 10% 이하
- 최종 alpha는 modal-trust fold의 25% 분위수
- 하나라도 실패하면 alpha=0으로 baseline 정확 복원

unit regret 분모에는 baseline unit MSE의 중앙값을 더해, baseline 오차가 거의
0인 unit에서 비율이 폭발하는 문제만 완화했다. 위험값 자체는 수축하지 않았다.

## 결과

### Stanford

- fold prior 채택률: 12.5%
- modal trust 합의율: 12.5%
- 거부 이유: fold prior consensus 부족
- v1.3 test: baseline과 동일
  - R² 0.0714
  - RMSE 81.444
  - raw mean/CVaR/max regret 모두 0

기존 fixed RBPR R² 0.0730의 소폭 이득은 validation unit을 바꾸면 유지되지
않았다. 따라서 강건성 기준에서는 이 이득을 포기하는 것이 맞다.

### ISU 250 mAh

- fold prior 채택률: 100%
- trust .02 합의율: 95.6%
- fold alpha IQR: 0
- OOF mean regret: -2.97%
- OOF CVaR regret: +3.98%
- OOF maximum regret: +12.78%
- 거부 이유: 최대 raw unit regret 10% cap 초과
- v1.3 test: baseline과 동일
  - R² 0.4798
  - RMSE 1.671
  - raw mean/CVaR/max regret 모두 0

ISU는 평균과 CVaR만 보면 prior가 안정적으로 보였지만 한 unit의 손상이
12.78%였다. hierarchical 수축에서 이 손상이 가려졌던 것과 일치한다.

## 판정

사용자가 요청한 “단일 성능 보정보다 안정성·강건성” 기준에서는
**Stability-First RBPR v1.3을 안전 실행 정책으로 채택**한다.

이 정책은 현재 두 retrospective 고호트에서 prior를 모두 거부한다. 따라서
“성능 개선 모형”이 아니라 “불안정한 개선을 배포하지 않는 executor”다.
fixed RBPR의 소폭 성능은 연구용 ablation으로 남기고, 운영·확증 경로에서는
v1.3의 exact baseline fallback을 사용한다.

전체 데이터셋 범용화는 v1.3 위에 contract-conditioned expert를 붙이는 CRPE
구조로 진행한다. v1.3은 모든 expert family가 공통으로 거쳐야 할 raw-risk
안전층이다.

## 검증

```bash
python experiments/stability_first_rbpr_v13_development.py
PYTHONPATH=src pytest -q tests/test_stability_first.py
```

결과 artifact:
`results/stability_first_rbpr_v13_development/results.json`
