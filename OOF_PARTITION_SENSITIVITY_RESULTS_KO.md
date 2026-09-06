# HUST OOF partition 민감도

Outer unit fold와 uncertainty-head seed를 다섯 번 바꿔 전체 OOF target을 다시 만들었다. 기존 PP 5개는 동일하게 고정했다.

| OOF split seed | 선택 γ | single PP 평균 pooled R² |
|---:|---:|---:|
| baseline PP | — | 0.7236 |
| 911 | 8 | 0.7306 |
| 912 | 0 | 0.7236 |
| 913 | 4 | 0.7315 |
| 914 | 4 | 0.7314 |
| 915 | 8 | 0.7374 |

평균은 `0.7309±0.0044`이고 5개 partition 중 4개가 기존 PP를 의미 있게 개선했다. 그러나 이전 단일 구성의 `0.7476`은 반복 결과보다 높아 대표값으로 쓰면 안 된다. 논문에는 partition 평균을 보고해야 한다.

이 실험은 head 효과가 완전히 우연은 아니라는 증거를 주지만, unit-bootstrap CI가 0을 포함하고 개선 크기도 partition에 민감하다. OOF head는 여전히 보조 모듈이다.

