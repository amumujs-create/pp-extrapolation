# PP-X 거리별 prior-set consensus 개발 실험

## 결론

현재 구현은 **PP-X 최종 모델로 승격하지 않는다**. 거리별 승인 집합,
disagreement envelope, exact fallback은 의도대로 작동했지만, 물리 unit을
하나씩 제외한 검증에서 안정적으로 승인되는 prior 집합이 형성되지 않았다.

이 결과는 “여러 prior를 섞으면 일반 외삽이 자동으로 강해진다”는 근거가
아니다. 현재 PP-X를 유지하고 이 구조는 기각된 challenger로 기록한다.

## 무엇을 시험했나

- prior-off fallback: trust 0 PP-X portfolio
- 후보 continuation: trust .02, .05, .10, .20, .40
- train support로부터의 거리 기준 0–50%, 50–80%, 80–100% shell
- shell별 physical-unit 평균 개선과 worst-20% harm budget을 모두 통과한
  후보만 승인
- 승인 후보의 중앙값을 예측값으로 사용
- validation에서 정한 90% disagreement envelope를 벗어나면 exact fallback
- 모든 validation unit은 자신을 제외하고 만든 정책으로 평가
- global set, 자유 shell set, 멀어질수록 집합이 줄어드는 nested set 비교

프로토콜은 실행 전에
`protocols/PRIOR_SET_CONSENSUS_DEVELOPMENT_PROTOCOL.md`로 동결했다.

## 결과

### Stanford

- fallback: pooled R² 0.0714, RMSE 81.4444, unit-macro R² 0.2391
- validation-best single trust .02: R² 0.0742, RMSE 81.3219
- 기존 continuous portfolio: R² 0.0714
- global/free/nested prior set: 모두 R² 0.0714, fallback 사용률 100%
- cross-fit 8/8 fold에서 승인 집합이 비어 있었다.

### ISU 250 mAh

- fallback: pooled R² 0.4798, RMSE 1.6712, unit-macro R² -1.8509
- validation-best single trust .05: R² 0.5205, RMSE 1.6044,
  unit-macro R² -1.6940
- 기존 continuous portfolio: R² 0.4917, RMSE 1.6520,
  unit-macro R² -1.6828
- global/free/nested prior set: 모두 R² 0.4798, fallback 사용률 100%
- global/nested는 cross-fit 45/45 fold에서 집합이 비었다.
- free-shell은 단 1/45 fold의 중간 shell에서만 trust .02를 승인했으며,
  전체 OOF 이득은 -0.05%였다. 따라서 최종 정책은 기각되었다.

## 해석

실패 원인은 prior-set 원리 자체의 반증이라기보다, 이번 후보들이 충분히
이질적인 prior가 아니었다는 데 있다. 다섯 후보는 서로 다른 물리 가정이
아니라 같은 prior-residual continuation의 trust 강도 변형이다. 따라서
후보 간 일치는 독립적인 구조 합의가 아니며, unit을 하나 제외하면 개선
근거도 사라졌다.

ISU에서 trust .05 단일 후보의 test 수치는 좋아졌지만, 이것만으로
prior-set architecture를 주장하면 안 된다. 동일 후보가 cross-fit
physical-unit risk 조건을 안정적으로 통과하지 못했기 때문이다.

## 논문 판단

- 현재 PP-X 정의와 주장은 변경하지 않는다.
- 이 실험을 main positive result나 novelty 근거로 사용하지 않는다.
- “거리별 multi-prior consensus가 검증됐다”고 쓰지 않는다.
- 필요하면 supplement의 negative development result로 공개할 수 있다.

후속 연구에서 다시 시험하려면 affine boundary, monotone/history,
mechanistic law처럼 **서로 다른 구조 가정에서 유도된 frozen prior**가
최소 3개 필요하다. 그 전에는 trust 변형을 multi-prior ensemble로
명명하지 않는 것이 정확하다. 새 후보를 만든 뒤에도 동일한 unit-cross-fit
승인 규칙과 미개봉 cohort 확인을 유지해야 한다.
