# NASA Milling locked PP replay — FAIL with certified abstention

Protocol commit: `ebc2e8a`, created before result execution.

| metric | validation material 1 | source material 2 |
|---|---:|---:|
| PP pooled R² | +0.497 | -4.826±0.000 |
| plain NN pooled R² | — | -15.462±19.427 |

사전 성공 조건인 `PP mean pooled R² > 0`을 만족하지 못했으므로 predictive
transfer는 **FAIL**이다. 결과를 보고 split이나 threshold를 바꾸지 않았다.

수치 validation만 사용하면 PP를 잘못 승인한다. 반면 predeclared mechanism
context인 material을 포함한 applicability certificate는 train level `{1}`과
source level `{2}`의 overlap이 0이므로 label 없이 **ABSTAIN**한다. 따라서 이
실험은 PP의 범용 성공 근거가 아니라, typed regime coverage 검사가 numeric
validation gate보다 먼저 필요하다는 실패 사례다.

이 결과로 방어 가능한 기여는 정확도 향상이 아니라 다음 정책이다.

`compile observable roles -> certify mechanism coverage -> validate predictor -> predict/abstain`

NASA Milling 표본은 train 33, validation 4, source-tail 10행으로 매우 작고 이전
PAE 개발에서 데이터가 관찰됐다. 독립 prospective 성공으로 세지 않는다.
