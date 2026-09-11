# Risk-Budgeted Prior Residual 개발 실험

작성자: 박진서  
상태: 이미 개봉한 Stanford·ISU retrospective 개발 결과이며 신규 확증이 아니다.

## 질문

prior가 거부될 때 약한 fallback으로 무너지는 문제를 피하면서, prior가 유효할
때는 연속적으로 이용할 수 있는가?

## 고정 모형

\[
\hat y(x)=B(x)+\alpha\,m(x)\{P(x)-B(x)\}
\]

- \(B\): trust-zero 4개 구조 × 5 seed의 강건 평균
- \(P\): 양의 trust별 4개 구조 × 5 seed 평균
- \(m(x)=1/(1+d(x)+u(x))\): train support 밖 거리와 prior-residual
  seed disagreement가 커질수록 감소
- \(\alpha\): validation physical-unit 기준 mean excess MSE와 worst-20%
  excess-loss CVaR가 baseline의 2% 이내인 최대값
- 제약이 성립하지 않으면 \(\alpha=0\), 따라서 baseline을 수치적으로 정확히 복원

## 핵심 결과

### Stanford

- baseline portfolio: R² 0.0714, RMSE 81.444
- 기존 continuous portfolio: R² 0.0714, RMSE 81.444
- 최종 local-CVaR: trust .02, alpha 1.00, 실제 평균 prior 계수 .338,
  R² 0.0730, RMSE 81.373
- unconstrained prior: R² 0.0742, RMSE 81.322

최종안은 baseline보다 소폭 개선했지만 효과 크기는 매우 작다.

### ISU 250 mAh

- baseline portfolio: R² 0.4798, RMSE 1.671
- 기존 continuous portfolio: R² 0.4917, RMSE 1.652
- 최종 local-CVaR: trust .05, alpha .52, 실제 평균 prior 계수 .177,
  R² 0.4866, RMSE 1.660
- unconstrained prior: R² 0.5205, RMSE 1.604

최종안은 baseline보다 개선했지만 기존 continuous portfolio보다 평균 성능이
낮다. 반면 held-out unit의 baseline 대비 worst-20% excess-loss CVaR는
continuous portfolio 5.96%, unconstrained prior 11.41%, local-CVaR 2.46%였다.
validation에서 고정한 2% 예산을 test에서 0.46%p 초과했으므로 test 보증으로
부르면 안 되지만, 위험 감소 방향은 명확했다.

## 구조 ablation

- ISU에서 support만 사용: R² 0.4851
- seed disagreement만 사용: R² 0.4886
- 둘 다 사용: R² 0.4866

seed disagreement가 support distance보다 유용했고, 두 신호의 단순 가산 결합은
최선이 아니었다. 따라서 현재 \(m(x)\) 함수는 추가 개선 대상이다.

epsilon=0에서는 두 고호트 모두 alpha=0으로 baseline을 정확히 복원했다.
ISU에서 epsilon .01/.02/.05의 R²는 각각 0.4833/0.4866/0.4923이었다.
단, .05를 이 결과를 보고 채택하면 post-test 튜닝이므로 신규 확증 전에
별도 protocol로 동결해야 한다.

## 판정

이 구조는 “강한 baseline + sample별 prior residual + physical-unit CVaR
risk budget”이라는 논문 후보의 모형적 주장은 만든다. 그러나 현재 결과만으로
완성 모형이나 노벨티를 확정할 수는 없다.

1. prior 거부 시 강한 baseline의 정확 복원: 통과
2. 두 고호트에서 baseline 비악화: 통과
3. 기존 continuous portfolio보다 명확한 평균 성능 향상: 실패
4. 위험-성능 trade-off의 실증: 통과

따라서 RBPR은 **안전성 중심 연구 후보**로 유지하되 v1.2 최종 채택은 보류한다.
다음 단계는 단순 \(1/(1+d+u)\) 대신 group-OOF로 단조 calibration한 risk
budget을 고정하고, 완전히 미개봉인 고호트에서 한 번만 확인하는 것이다.

재현:

```bash
python experiments/risk_budgeted_prior_residual_development.py
PYTHONPATH=src pytest -q tests/test_risk_budgeted_prior.py
```
