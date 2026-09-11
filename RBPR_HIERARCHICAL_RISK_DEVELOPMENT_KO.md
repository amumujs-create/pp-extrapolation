# RBPR 계층형 위험 부분수축 실험

작성자: 박진서  
상태: 이미 개봉한 Stanford·ISU retrospective 개발 결과이며 확증이 아니다.

## 고정 가설

strict-tail 행이 1~3개뿐인 physical unit의 raw MSE가 과도하게 흔들리는 문제를
줄이기 위해 paired unit excess loss를 다음처럼 부분수축했다.

\[
e_g^H=w_g e_g+(1-w_g)\bar e,\qquad
w_g=\frac{n_g}{n_g+5}
\]

평균위험은 raw unit excess를 그대로 사용하고 worst-20% CVaR에만 수축값을
사용했다. 기본 `n0=5`는 결과 전에 protocol로 고정했다.

## 결과

### Stanford

- baseline R² 0.0714
- 기존 fixed RBPR R² 0.0730
- hierarchical full RBPR R² 0.0730
- test raw CVaR -0.069%, hierarchical CVaR -0.071%

유의미한 추가 개선은 없지만 위험도 증가하지 않았다.

### ISU 250 mAh

- baseline R² 0.4798
- 기존 fixed RBPR R² 0.4866
- hierarchical full RBPR R² 0.4740
- unit-평균 excess risk -2.58%
- test raw CVaR +12.29%
- test hierarchical CVaR +5.39%

평균 physical-unit loss는 개선됐지만 pooled R²와 tail safety가 악화됐다.
validation OOF에서 hierarchical CVaR는 +0.51%로 2% 예산을 통과했으나,
test raw CVaR는 +12.29%였다. 부분수축 인증이 실제 tail 손상을 가렸다.

## 수축강도 ablation

- `n0=0`: ISU prior 거부, baseline R² 0.4798
- `n0=2`: ISU prior 거부, baseline R² 0.4798
- 사전 고정 `n0=5`: trust .02 채택, R² 0.4740, raw CVaR 12.29%
- `n0=10`: trust .05 채택, R² 0.5205, raw CVaR 11.41%

`n0=10`은 평균 성능만 보면 매우 좋아 보이지만 hierarchical CVaR는 -3.71%로
표시된다. 이는 강한 수축이 위험 unit을 전체 평균 속에 숨기는 반례다.
이 값을 보고 `n0=10`을 채택하면 post-test 튜닝일 뿐 아니라 안전성 주장도
잘못된다.

## 판정

계층형 부분수축을 **단독 risk certificate로 사용하는 안은 기각**한다.
실패 결과는 “소표본을 무시하거나 과도하게 수축하면 평균 성능은 좋아 보여도
tail failure가 가려진다”는 강한 ablation 증거로 보존한다.

현재 순위:

1. 안전성 후보: 기존 fixed RBPR — ISU R² 0.4866, raw CVaR 2.46%
2. 평균성능 참고 상한: unconstrained trust .05 — R² 0.5205, raw CVaR 11.41%
3. hierarchical RBPR — 채택 불가

다음 모형은 부분수축 평균만 쓰면 안 된다. raw-CVaR hard cap과 hierarchical
CVaR를 동시에 만족시키는 dual certificate 또는 소표본 posterior
upper-confidence bound가 필요하다. 이 고호트의 test를 반복해 기준을 고르면
안 되므로 수치 기준은 신규 미개봉 고호트 전에 동결해야 한다.
