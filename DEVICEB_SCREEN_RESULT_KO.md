# Device-B 경계형 외부 고호트 적격성 결과

## 판정: inconclusive

프로토콜 커밋 `d257e12` 뒤 공식 `Auburngrads/SMRD.data`의
`deviceb.rda`를 처음 열었다.

- 파일 크기: 4,002 bytes
- SHA-256: `960deef1c52c19747a959e7fd765cb6d57603dcdccf6e7d43bea273439ce815f`
- 전체: 570행, 34개 장치, 3개 가속 온도
- 문헌 고장 정의: 초기 출력 대비 0.5dB 초과 power drop
- 경계 도달 장치: 27개
- 경계 도달 전에 최소 8회 관측된 장치: `110`, `112`, `116`, `118`
- 이 네 장치의 pooled tail: 13행

사전 최소 기준은 8개 적격 장치와 32개 tail 행이므로 학습·성능평가 없이
inconclusive로 종료했다. 높은 온도의 장치는 0.5dB 경계를 매우 빨리 넘어서
외삽 history가 부족하고, 낮은 온도의 장치는 시험 종료 전에 경계에 도달하지
않았다. 이는 작은 accelerated-degradation 자료에서 흔한 event/history
trade-off다.
