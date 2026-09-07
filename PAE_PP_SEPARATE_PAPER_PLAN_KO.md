# PAE와 PP 별도 논문 분리안

## 원칙

PAE와 PP는 서로를 실행 의존성으로 두지 않는다. 코드, 입력, 가설, 주 비교표와 핵심
기여를 분리한다. 박사논문에서만 두 연구를 외삽 생명주기의 연속 단계로 연결한다.

## Paper A: PAE

PAE의 연구 질문은 도메인에서 관측 가능한 정보로 어떤 구조 prior를 모델에 허용하고,
그 강도를 어떻게 선택할 것인가이다. 핵심 기여는 typed observation contract, admissible
prior compilation, hard boundary와 direction, prior-off routing이다.

PAE 논문에서는 seed-consensus calibration, output error taxonomy, validation-unit correction
transport와 shift-ray 기반 output calibration 승인을 제외한다.

## Paper B: PP

PP의 연구 질문은 strict out-of-support RUL에서 학습된 neural predictor의 출력 오류가
새로운 physical unit과 외삽 위치로 전이 가능한지를 test label 없이 어떻게 판정할
것인가이다.

PP는 train/validation/test causal feature, physical unit ID, 데이터 프로토콜에서 사전
선언한 extrapolation coordinate, train/validation label과 test prediction만 사용한다.
PAE contract, PAE compiler, PAE model 또는 PAE가 생성한 prior는 필요하지 않다.

PP의 핵심 기여는 다음으로 제한한다.

1. affine-tail plus neural residual PP backbone;
2. group-LOO output error diagnosis;
3. exact seed replication과 physical-unit bootstrap;
4. validation/test shift-ray compatibility;
5. correction transport, identity fallback 및 별도 applicability abstention.

PP 논문에서는 도메인별 물리식 생성, hard-boundary integral architecture, LLM/contract 기반
prior compilation과 관측 가능한 prior 종류의 자동 선택을 제외한다.

## 중복을 피하는 표현

| 피할 표현 | PP 논문 표현 |
|---|---|
| PAE가 컴파일한 축 | predeclared extrapolation coordinate |
| prior compiler | error-transport verifier |
| domain contract routing | geometry eligibility test |
| hard physical constraint | bounded shape-preserving output map |
| 새 도메인 prior 선택 | correction transport approval |

## 박사논문에서만 연결할 내용

박사논문의 종합 장에서는 두 논문을 다음 순서로 연결한다.

`PAE: assumption selection before training → PP: error transport after training`

PAE는 PP 없이 실행되고 PP는 PAE 없이 실행된다. 박사논문은 두 독립 연구가 외삽 모델의
서로 다른 실패 지점을 다룬다는 상위 프레임만 제시한다.

## 독립 논문 제목 후보

- PAE: **Typed Prior Compilation for Cross-Domain RUL Extrapolation**
- PP: **Geometry-Compatible Replicated Error Transport for Out-of-Support RUL Prediction**

PP 제목에서 PAE, contract compilation, physics-informed라는 표현을 사용하지 않는다.
