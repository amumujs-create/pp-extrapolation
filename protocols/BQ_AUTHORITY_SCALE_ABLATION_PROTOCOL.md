# Residual authority scale ablation — Fixed / Dual / 3-scale / Continuous

작성자: 박진서  
상태: matched development ablation (논문 Algorithm 1 동결 변경 아님)

## 질문

Dual-scale이 “2개가 최적”이라서가 아니라, **고정 bound의 MICH 반례를 최소 자유도로 복구**한 것이라면,
같은 residual network에서

- Fixed \(B=2\)
- Dual \(B_L=2, B_H=6\) (frozen)
- 3-scale ladder \(B\in\{2,4,6\}\)
- Continuous \(B(X)\in[2,6]\)

를 비교하면 무엇이 이기는가?

## 프로토콜

- 데이터: Sunwoda · RWTH · MICH joint battery split (dual-scale replay와 동일)
- seeds 42–46
- train/validation으로 epoch 선택 → full refit → source score
- 네 arm은 width / optimizer / affine prior / support feature를 공유
- arm 간 선택은 **validation macro MSE만** 사용 (test peeking 금지)

## 주장 경계

- dual이 최적 개수라고 주장하지 않는다.
- 3-scale / continuous가 더 좋으면 “일반화 여지”로 기록한다.
- validation이 dual을 고르지 않거나 RWTH가 더 나빠지면, 복잡도 증가의 위험을 기록한다.
