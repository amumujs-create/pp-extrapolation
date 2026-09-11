# RBPR 교차적합 단조 보정 실험

작성자: 박진서  
상태: 이미 본 Stanford·ISU 자료를 이용한 retrospective 개발 실험

## 개선 시도

기존 고정식 `1/(1+d+u)` 대신 다음 단조 budget을 physical-unit OOF로
학습했다.

\[
m(x)=\exp\{-k_d d(x)-k_u u(x)\},\qquad k_d,k_u\ge 0
\]

거리 감쇠, seed-disagreement 감쇠, alpha, prior trust를 test label 없이
validation group OOF에서 선택했다. OOF mean excess MSE와 worst-20% CVaR는
baseline의 2% 이내여야 하며, 만족하는 후보가 없으면 alpha=0이다.

## 결과

### Stanford

- baseline R² 0.0714
- 기존 고정 RBPR R² 0.0730
- crossfit global R² 0.0742
- distance-only/full R² 0.0730
- disagreement-only R² 0.0716

crossfit global이 가장 높았지만 개선 폭은 작다. full 모형은 거리 감쇠 4,
불확실성 감쇠 0을 선택해 기존 RBPR과 사실상 같은 수준이었다.

### ISU 250 mAh

- baseline R² 0.4798
- 기존 고정 RBPR R² 0.4866
- crossfit global R² 0.4787
- disagreement-only R² 0.4787
- distance-only/full: prior 거부, baseline 정확 복원(R² 0.4798)

교차적합 모형은 기존 RBPR을 개선하지 못했다. global은 trust .40,
alpha .07이라는 지나치게 작은 경로를 선택했고 test CVaR도 validation 예산
2%에서 2.85%로 상승했다.

## 원인

ISU validation tail은 선언된 46개 중 실제 tail 행이 있는 45개 physical
unit에 167행뿐이며, 24개 unit이 3행 이하이다.
각 unit MSE와 worst-20% CVaR를 다시 leave-one-unit-out하면 위험 순위가 매우
불안정하다. 단조 함수의 자유도를 늘리자 유용한 trust .05가 탈락하고,
작은 alpha의 trust .40 또는 완전 fallback이 선택됐다.

따라서 “gate를 더 복잡하게 만들면 개선된다”는 가설은 이 실험에서 기각됐다.
현재 데이터 규모에서는 기존 고정 RBPR이 더 낫다.

## 판정과 다음 설계

- crossfit monotone RBPR: 채택하지 않음
- 기존 2% fixed RBPR: 안전성 후보로 유지
- 실패 결과와 exact fallback은 ablation 증거로 보존

다음 개선은 test를 다시 보며 감쇠계수를 조정하는 방식이면 안 된다. 사전에
고정한 hierarchical risk estimator가 필요하다. 특히 행이 1~3개인 unit의
excess loss를 전체 평균으로 부분수축하고, 충분한 unit만 CVaR tail에 직접
반영하는 방식이 합리적이다. 이 변경은 새로운 미개봉 고호트 전에 별도
protocol로 동결해야 한다.
