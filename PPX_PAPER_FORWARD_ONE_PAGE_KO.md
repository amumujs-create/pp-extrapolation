# PP-X Final v4 — 한 장

작성: 박진서
정본: [`PPX_FINAL_PAPER_VERSION_KO.md`](PPX_FINAL_PAPER_VERSION_KO.md)
Manifest: [`protocols/PPX_FINAL_PAPER_V4_MANIFEST.json`](protocols/PPX_FINAL_PAPER_V4_MANIFEST.json)

## 로직

```text
train-only contract audit
  (group/time/regime user key first;
   blank → unambiguous train-only detection;
   never x[:,0] as time)
→ contract-computable PP-X candidates
→ family aggregate validation MSE argmin
→ seed/fold 내부 configuration validation MSE argmin
→ 선택 고정
→ test prediction ensemble
```

- 데이터셋 이름·registry·test는 선택 입력이 아니다.
- 시간 후보가 없거나 여러 개면 ordered progression/history 후보는 OFF다.
- regime은 사용자 키 우선, 빈칸이면 train 이산성과 unit 안정성으로 검사한다.
- unit ID는 사용자 키 우선, 빈칸이면 반복 ID·연속 블록으로 검사하며 모호하면 입력을 요구한다.
- BQ는 강제가 아니라 affine와 함께 열 수 있는 prior 옵션이다.
- Direct는 PP-X 후보가 아니라 외부 대조군이다.
- Test를 본 뒤 route를 바꾸지 않는다.

## 그림 재현

| Setting | PP-X pooled R² |
|---------|---------------:|
| HUST | 0.958 |
| Sunwoda | 0.939 |
| N-CMAPSS | 0.937 |
| Virkler | 0.888 |
| RWTH | 0.878 |
| MATR-b2 | 0.862 |
| MICH | 0.751 |
| NASA | 0.584 |
| MATR2019 | 0.466 |

- 그림 반올림 값 재현 **9/9**
- 양의 R² **9/9**
- Macro mean R² **0.807**
- 최강 동일예산 baseline 대비 **8/9** 우세

상세 결과:
[`PPX_FINAL_RESULT_REPRODUCTION_KO.md`](PPX_FINAL_RESULT_REPRODUCTION_KO.md)

## 재현

```bash
PYTHONPATH=src:experiments:../ca-css-ncmapss \
  python experiments/ppx_final_result_reproduction_v1.py
```

주의: 기존 개발 cohort의 retrospective reproduction이며 prospective 증거가 아니다.
