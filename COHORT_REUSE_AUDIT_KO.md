# 기존 보존 코호트의 v2 재사용 가능성 감사

사전등록 v2는 결과를 아직 보지 않은 test target에만 적용할 수 있다. “이전에 개발에 쓰지 않았다”는 조건만으로는 충분하지 않으며, prediction·metric·label이 이미 materialize된 코호트를 같은 v2의 confirmatory test로 재사용할 수 없다.

| 코호트 | 기존 상태 | v2 새 확증에 재사용 | 이유 |
|---|---|---|---|
| MATR 2019-01-24 | 사전고정 latent-transition PP 평가 완료 | 불가 | pooled R² 0.257 결과가 이미 공개됨 |
| MATR batch2 2017-06-30 | 봉인 confirmatory 평가 완료 | 불가 | 이후 최종 PP 개발에도 사용됨 |
| Axial fan sealed | 세 configuration outcome 평가 완료 | 불가 | sealed RUL 세 파일이 모두 열림 |
| Na-ion group 1–6 | route/gate 개발 또는 prospective 평가 완료 | 불가 | unit route와 pooled 결과가 이미 materialize됨 |
| Zn-ion confirmation files | 여러 one-shot 평가 및 provenance audit 완료 | 불가 | duplicate/eligible-unit 결과가 이미 확인됨 |
| HNEI/CALCE/Matwi | locked/external 결과 평가 완료 | 불가 | test outcome 확인 및 일부 protocol repair가 발생함 |

따라서 로컬에 남은 “sealed” 파일은 **재현성 검증과 ablation**에는 사용할 수 있어도, v2의 새 prospective success에는 사용할 수 없다. 이를 새 성공처럼 선언하면 data reuse가 된다.

다음 확증은 (a) 별도 source에서 내려받은 코호트, 또는 (b) 기존 공개 archive 안에서 지금까지 한 번도 추출·feature화·label 열람하지 않은 unit 묶음이어야 한다. 후자의 경우에도 unit manifest와 SHA-256을 v2 protocol commit 전에 작성해야 한다.
