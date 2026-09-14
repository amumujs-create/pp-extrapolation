# PP-X prior gate 버전

작성자: 박진서

| 버전 | 기본? | 실제 분기 | 상태 |
|---|---|---|---|
| `final` | 예 | 경계 있음 → BQ, 없음 → affine | PP-X Final. 계산하지 않는 OOF/group/mode 검사 없음 |
| `v1_declared` | 아니오 | 경계 없으면 group/OOF/mode 칸을 검사 | 보관용. DS03·FEMTO 역사 감사만 |

코드:

- `src/pp_extrapolation/transferability_gate.py`
- 상수 `PRIOR_GATE_VERSION = "final"`
- 보관 함수 `select_ppx_route_v1_declared`

Final 9-setting 결과와 Final ablation은 `final` 경로다. `v1_declared`를 Final 게이트라고 쓰지 않는다.

Ablation:

- 9-setting core/executor on/off는 Final. 재학습하지 않는다.
- FEMTO prior abstention은 `v1_declared` 보관. Final 표의 BH 보정에 넣지 않는다.
