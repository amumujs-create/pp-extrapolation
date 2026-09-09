# MEMSS Fatigue 외부 고호트 적격성 결과

## 판정: inconclusive

수치 데이터를 열기 전에 커밋 `b438ff7` 및 그 pre-download tuning amendment로
프로토콜과 분할을 고정했다. 공식 미러의 `nlme/Fatigue.csv`를 받은 뒤 처음으로
적격성을 확인했다.

- 파일 크기: 4,964 bytes
- SHA-256: `6838417e55759d496084e2b82319af4049d4cf88ade4db22f8f1936b463d4a98`
- 전체: 262행, 21개 독립 시험편
- 사전 고정 test: `7`, `10`, `21`, `17`
- 문헌 고장경계 `relLength = 1.60 / 0.90`을 관측으로 bracket한 test:
  `7`, `10`
- right-censored test: `21`, `17`

사전 규칙은 적격 test 시험편이 3개 미만이면 성능을 계산하지 않고
inconclusive로 판정하도록 했다. 실제 적격 시험편은 2개이므로 PP/MLP 성능을
돌리지 않았고 이 데이터는 성공 또는 실패 근거에 포함하지 않는다.

이 결과는 모델 실패가 아니라 작은 데이터에서 임의 holdout과 event scarcity가
충돌한 사례다. 이후 소형 고호트는 수치를 보기 전에 leave-one-unit-out 평가를
고정해 적격 장치를 낭비하지 않는 설계를 사용한다.
