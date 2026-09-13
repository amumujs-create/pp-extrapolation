# OOF threshold-policy robustness (official tuning evidence)

작성자: 박진서  
상태: **공식 threshold 튜닝의 근거 실험**  
상위 프로토콜: `protocols/PPX_THRESHOLD_TUNING_OOF_PROTOCOL.md`

## 목적

고정 threshold `(2%, 60%, 1.10)`를 임의값이 아니라,
**미리 선언한 합리적 격자 + unit holdout OOF 안정성**으로 고르고 동결하기 위한 튜닝 증거다.

Algorithm 1 실행 루프는 바뀌지 않는다. 바뀌는 것은 “τ를 어떻게 정하는가”뿐이다.


## 하지 않는 것

- test에서 threshold를 고르지 않는다
- OOF win을 최대화하는 τ*를 새 default로 올리지 않는다
- Algorithm 1 구조를 바꾸지 않는다

## 하는 것

후보

$$ \tau_g\in\{1,2,5\%\},\; \tau_w\in\{50,60,70\%\},\; \tau_r\in\{1.05,1.10,1.20\} $$

각 셀에 대해 validation unit leave-one-out으로

- OOF executor win fraction
- OOF mean / worst policy regret
- Route stability (LOO 결정 = full-val 결정)

을 계산한다.

## 해석 규칙

“합리적인 threshold 영역 중에서 unit holdout에 대해 승인 결정이 반복적으로 유지되고 worst-case regret이 작은 보수적 정책을 선택/유지했다.”

최종 배포 스토리:

Development OOF → Freeze τ=(2%,60%,1.10) → 새 데이터셋에서 τ 재조정 금지 → 그 셋 Val로 executor 승인 → Test untouched.
