# Regime-scale PP 사전 확증 v1 감사

사전 동결 commit `27c340e`이 난 뒤 공식 Zn-ion test 4–6번 셀을 받아 한 번 실행했다.

| 모델 | pooled R² | unit-macro R² |
|---|---:|---:|
| plain MLP, seed mean | -0.539 | -18.594 |
| regime-scale PP, seed mean | **0.342** | **0.512** |
| regime-scale PP, seed median | **0.669** | **0.684** |

사전 수치 기준은 통과했지만, **독립 확증 성공으로 계산하지 않는다.** 원본 SHA-256 감사에서 `439-1`의 xlsx가 이전에 사용한 `438-3`과 byte-for-byte 같은 파일로 확인됐다.

- SHA-256: `18c7a7009c9bd59a26a0f45c6df1fa6d83cae86475c424da801aba78c3fd485e`
- 용량 궤적, EOL, 예측과 지표가 이전 셀과 완전히 같았던 이유다.
- `412-2`는 EOL이 152-cycle 경계보다 빨라 평가 tail이 없었다.
- 실제로 새롭고 평가 가능한 셀은 `410-2` 하나였고 tail도 14행이었다. 이 셀에서 PP R²는 모든 seed에서 약 **0.716**으로 양수였지만 확증 표본으로는 부족하다.

따라서 이 실험은 `사전 성공 수치는 통과, 소스 중복으로 독립 확증은 무효`로 기록한다. 다음 확증은 모델 실행 전에 원본 hash로 모든 기존 셀과의 중복을 제거하고, EOL 뒤 최소 tail 길이를 사전 eligibility로 동결해야 한다.
