# Oxford untouched protocol feasibility 결과

Protocol `c92622a`를 공개 고정한 후 Oxford Battery Degradation Dataset 1을 처음 받아
실행했다. 고정된 25th-percentile capacity cutoff는 `585.746 mAh`였다.

| split | 고정 조건을 만족한 8-step windows |
|---|---:|
| Train Cell1--Cell5 | 213 |
| Validation Cell6 | **1** |
| Source Cell7--Cell8 | 24 |

Protocol은 validation 또는 source가 8개 미만이면 중단하도록 미리 정했다. Validation이
1개뿐이므로 model fitting, shell calibration, test-label scoring을 하지 않고
`INFEASIBLE`로 종료했다. Cell이나 cutoff를 결과에 맞춰 바꾸지 않았다.

이는 성능 성공도 실패도 아니며 untouched accuracy 표에 넣지 않는다. 대신 한 cell의 먼
capacity tail만으로 shell을 보정하기 어렵다는 설계 한계를 보여 준다. Oxford를 다시 쓰려면
새로운 protocol과 별도의 confirmatory dataset이 필요하며, 해당 결과는 development로
분류해야 한다.
