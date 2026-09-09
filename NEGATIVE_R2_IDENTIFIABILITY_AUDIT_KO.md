# XJTU·FEMTO 음수 R2 식별가능성 감사

> 재감사 정정: 아래 통계는 특정 모델군의 한계일 뿐 일반적인 식별 불가능성 증명이 아니다. FEMTO 학습 파일과 캐시를 대조한 결과 로더가 진동 채널 대신 첫 두 시간 열을 사용한 오류가 확인됐다. 입력 수정 전 FEMTO의 성능 한계 주장은 보류한다. 자세한 내용은 `MODELING_IMPROVEMENT_REAUDIT_KO.md` 참조.

이 감사는 test를 이용해 모델을 더 선택하지 않고, 왜 structural PP가 경쟁모델을 넘은 후에도 pooled R2가 음수인지 수치화한다.

## XJTU

- train RUL 평균 210.5, validation 64.2, test 1,072.4
- train 최대 RUL 525보다 큰 test 행: **70.7%**
- 예측을 train target 범위에 완벽히 맞추어도 가능한 oracle R2: **-0.516**
- test label을 본 공유 총수명 oracle의 최고 R2도 **0.231**

따라서 R2 0.3 이상은 하나의 condition-level scale만으로도 불가하고 bearing별 lifetime scale이 필요하다. 현재 input에서 train/validation은 운전조건별 수명 평균이 123·313인데 test는 1,407이므로, 세 번째 조건의 scale을 정하는 라벨 또는 물리 load-life law가 없다.

## FEMTO

- 학습 run-to-failure bearing 5개, validation bearing 1개, test endpoint 11개
- test endpoint RUL 범위 339--7,570 s, 평균 2,855 s, 표준편차 2,631 s
- 같은 condition 1의 RUL CV 0.84, condition 2의 CV 0.91

즉 condition과 elapsed time이 같아도 remaining-life scale이 큰 폭으로 달라진다. Causal prefix PP가 -1.165에서 -0.571로 오른 것은 유효하지만, 양의 R2를 위해서는 추가 run-to-failure bearing, bearing-specific manufacturing covariate, 또는 풍부한 raw vibration pretraining으로 lifetime-scale head를 식별해야 한다.

## 결론

현재의 정직한 최종값은 XJTU -0.843, FEMTO -0.571이다. 둘 다 관측된 경쟁모델보다 높지만 성공으로 세지 않는다. 동일 입력으로 양수 개선이 불가능하다는 증거는 확보되지 않았다. 특히 센서 입력 오류를 먼저 수정하고 재평가해야 한다.
