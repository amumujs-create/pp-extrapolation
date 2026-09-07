# PP / FT 데이터셋별 비교 결과

동일 데이터셋 안에서 입력·분할·학습 한도·후보 수·seed 선택 규칙을 맞춘 사후 비교. 9개 후보를 seed42 validation으로 선택 후 seeds42–46 재학습. 최대150 epoch/patience25. 탐색 축과 계산량은 동일하지 않으며 최적 성능을 보장하는 튜닝이 아니다.

| 데이터 | PP 구조 | PP joint | FT-Transformer |
|---|---:|---:|---:|
| hust | -0.391 | -0.391 | -0.770 |
| virkler | -2.786 | -3.286 | -9.345 |
| matr2019 | -0.100 | -0.688 | 0.344 |

이번 조건에서 FT의 양의 pooled R²는 MATR에서만 확인됐다. PP도 HUST/Virkler에서 음수이므로 상대적 우위를 성공으로 부르지 않는다.

FT single-seed R²: HUST -1.352 ± 0.980; Virkler -21.643 ± 33.357; MATR 0.303 ± 0.037.

이전 HUST PP 0.811 및 MATR PP 0.257은 300 epoch와 다른 seed별 validation 설정 선택을 사용했다. 이번 공통 예산 결과와 섞지 않는다. 프로토콜 차이에 의한 성능 변화의 원인은 분리 실험 없이 단정할 수 없다.

HUST/Virkler는 seen-unit 후반 열화 구간, MATR는 unseen-unit capacity-tail이다. 데이터셋 효과와 외삽 유형 효과가 얽혀 있어 특정 데이터 속성이 원인이라고 결론내릴 수 없다.

HUST/Virkler 30개 저장 예측 재계산이 JSON과 일치했다. MATR는 이미 완료한 동일 프로토콜 결과를 재사용했다.
