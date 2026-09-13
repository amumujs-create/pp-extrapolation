# PP-X approval threshold sensitivity

작성자: 박진서  
상태: retrospective audit on frozen 12-setting common-backbone archive

## 질문

운영 임계값

$$ Gain=2\%,\quad UnitWin=60\%,\quad WorstRatio=1.10 $$

이 “잘 골라서”만 나온 결과인가, 근처 값에서도 false-accept 통제가 유지되는가?

## Grid

- Gain ∈ {1%, 2%, 5%}
- UnitWin ∈ {50%, 60%, 70%}
- WorstRatio ∈ {1.05, 1.10, 1.20}

총 27 cells. CI gate는 끄고, 논문에서 말하는 operational thresholds와 맞춘다.

## 비교 baseline

- Validation RMSE only
- Frozen 2/60/1.10

## 성공 기준

- frozen 근처에서 FA가 RMSE-only(4)보다 낮게 유지되면 “근처에서도 보수적 승인” 지지
- 전 grid에서 결과가 크게 요동하면 threshold fragility를 limitation으로 기록
