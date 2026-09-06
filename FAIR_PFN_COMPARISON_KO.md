# 동일 조건 PFN 비교

## C-MAPSS FD002+FD004 — 완료

- split: 기존 strict OP1–OP3 convex-hull 밖 final 2,616행
- 입력: 동일 robust 115 features
- PFN train cap: 3,000행
- seeds: 42–46
- 주 지표: raw pooled R²

| 모델 | pooled R² | seed별 값 |
|---|---:|---|
| PFN v3 | 0.607±0.029 | 0.634, 0.593, 0.634, 0.558, 0.616 |
| 기존 staged PP | **0.749±0.015** | 기존 동결 결과 |
| exact-hull support-adaptive PP | 0.747±0.011 | 0.736, 0.734, 0.761, 0.750, 0.757 |

Support-adaptive PP와 PFN 차이는 **+0.140**이며 PP가 PFN의 5개 seed를 모두
상회했다. 다만 새 gate는 validation에서 `beta=0`이 선택되어 기존 PP보다
향상되지 않았다. 따라서 C-MAPSS에서는 support 감쇠가 아니라 soft anchor만
작동했고, 기존 staged PP와 사실상 동률이다.

PFN seed 변화에는 모델 random state와 3,000행 random subsample 변화가 함께
포함된다. 이는 기존 PFN 실행기의 seed 정의를 그대로 따른 것이다.

## HUST·Virkler·NASA — 인증 대기

동일 입력, 동일 frozen split, 최대 3,000 equal-unit 행, seeds 42–46 실행 코드가
준비됐다. 공식 TabPFN cloud client의 Prior Labs 재로그인이 완료되면 실행된다.
