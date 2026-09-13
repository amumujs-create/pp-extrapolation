# DS03 prior-off fallback portfolio — validation selection protocol

작성자: 박진서  
상태: frozen equal-budget / PP-X DS03 artifacts 재사용 감사  
범위: **prior가 거절된 DS03 fallback만**. prior-on core는 변경하지 않는다.

## 질문

prior-off일 때 fallback을 `{direct NN, Engression, FT-Transformer, GroupDRO, ...}`
중에서 **validation만**으로 고르면, Engression에 진 약점이 줄어드는가?

## 왜 진 구간만 먼저 보나

맞다. 1차 테스트는 **진 구간(DS03)** 만으로 충분하다.

- prior-on 메인 9개에서 이미 우세한 경로를 건드리지 않는다.
- 가설이 “prior-off fallback이 약하다”이므로, prior가 실제로 거절된 DS03가
  가장 직접적인 반증/확증 지점이다.
- NASA 동률권은 후속 확인용으로 남겨도 된다.

## 누출 방지

- validation MSE는 각 모델이 30-candidate search에서 이미 고른 설정의
  validation 점수를 사용한다.
- test R²는 선택 후에만 읽는다.
- Engression을 test에서 이겼다고 골라면 안 된다.

## 성공 기준

- validation이 Engression 또는 FT를 고르고 test에서도 PP-X direct보다 개선되면
  fallback 강화 가설 지지.
- validation이 Engression을 고르지 못하면, “그냥 Engression으로 바꾸면 된다”는
  주장은 기각하고, fallback 선택 규칙 자체를 더 설계해야 한다.
