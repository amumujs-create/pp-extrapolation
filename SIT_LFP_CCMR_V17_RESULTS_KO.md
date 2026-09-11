# SIT LFP CCMR v1.7 순차 봉인 결과

## 판정

실제 50 Ah LFP 17-cell 자료에서 v1.7은 validation 위험을 검출해 test
전체를 exact persistence로 복원했다. 성능 성공은 아니다.

- 원본 ZIP: 4.3 GB
- range 추출한 cycle summary: 2.66 MB
- train/validation/test: 10/3/4 cells
- train/validation/test origins: 1,843/426/466
- train 진행도 최대 30%, test 최소 75%, strict OOD 100%

Validation에서 causal gate coverage는 5.16%였고 pooled/macro RMSE가
0.36%/0.30% 악화됐다. raw mean/CVaR/max regret도
0.55%/1.21%/1.21%로 인증 기준을 넘었다. 따라서 global veto가 test
전에 발동했다.

봉인 test에서는 R² 0.99423, RMSE 0.006230이지만 persistence와 정확히
동일하다. 개선과 raw regret는 모두 0이고 fallback replay error도 0이다.

## v1.8 개발에 준 정보

열린 test를 이용한 사후 진단에서 base CCMR 후보 자체는 pooled/macro
RMSE를 0.77%/3.15% 개선했고, raw unit mean/CVaR/max regret는
-16.03%/0%/0%, coverage는 47.00%였다. 즉 smooth degradation에서는
최근 다섯 번의 우연한 연속 승리만 요구하는 causal gate가 오히려
안정적인 보정을 제거했다.

이 결과는 v1.8의 validation-regime router를 설계하는 개발자료로만
사용하며 v1.8 성공 증거로 세지 않는다.

관련 파일:

- `protocols/SIT_LFP_SEQUENTIAL_CCMR_V17_PROTOCOL.md`
- `experiments/sit_lfp_sequential_ccmr_v17.py`
- `results/sit_lfp_sequential_ccmr_v17_amended/results.json`
- `protocols/CCMR_V18_REGIME_ROUTER_PROTOCOL.md`
