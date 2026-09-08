# 데이터셋별 문헌 SOTA와 최종 PP 비교 범위

## 원칙

문헌의 최고 숫자를 최종 PP pooled R²와 바로 비교하지 않는다. 현재 PP 평가는 unit/cell 분리와 health-tail 또는 operating-condition hull 밖을 강제한 strict extrapolation protocol이다. 문헌은 대체로 다른 train/test unit, 다른 관측 시점, 원시 신호 입력, RMSE·MAE·NASA score를 사용한다. 이 문서는 각 데이터셋에서 논문에 반드시 인용·재실행할 **대표 SOTA 계열**과 현재 직접 비교 가능성을 정리한다.

| PP 데이터셋 | 문헌의 대표 benchmark / 강한 모델 계열 | 문헌 지표·분할 | PP와 수치 직접 비교 | 논문에서 할 일 |
|---|---|---|---|---|
| HUST | BatteryML·BatteryLife의 Transformer, LSTM, CNN 및 feature-based RUL baseline | 초기 cycle/표준 RUL benchmark, 주로 MAE | 불가 | 동일 raw cell·현재 strict tail split에 Transformer 및 BatteryML feature baseline 재실행 |
| MATR2019 / MATRb2 | Attia의 battery-lifetime ML, BatteryML/BatteryLife의 Transformer·GP·feature baseline | early-life lifetime 또는 별도 batch split, MAE/RMSE | 불가 | 현재 30/9/9 tail split에 sequence Transformer 및 feature baseline 재실행 |
| Sunwoda / RWTH | BatteryLife·BatteryML 계열의 multi-dataset battery-life 모델 | 공통 battery-life protocol, 보통 early-cycle life 예측 | 불가 | cell-disjoint tail split을 유지하며 BatteryLife feature encoder를 같은 정보 예산으로 재실행 |
| NASA battery | CNN-GRU, CNN-LSTM, Transformer 계열 | B0005/6/7/18별 holdout·scarcity 조건, RMSE/MAE | 불가 | 현재 leave-one-battery-out split에 CNN-GRU/sequence Transformer를 동일 causal history로 재실행 |
| Virkler | Paris-law/NASGRO 계열 및 history-aware crack-growth NN | crack length 또는 growth-rate 예측, 하중/재료 조건별 | 대체로 불가 | 현재 crack-tail split에 Paris-law fitted baseline과 history NN을 같은 target으로 재실행 |
| N-CMAPSS | LSTM, GRU, TCN, Transformer/TSMixer | standard subdataset, RMSE와 NASA score | 불가 | 현재 hard TRA-hull split에 sequence Transformer·TCN을 same-window 입력으로 재실행 |
| XJTU-SY | raw-vibration CNN, TCN-LSTM, enhanced Transformer | condition별 bearing split, RMSE/score | 불가 | raw vibration을 쓴 sequence model과 현재 feature-level transport split은 별도 표로 제시 |
| FEMTO/PRONOSTIA | CNN/LSTM/Transformer 및 transfer prognostics | bearing별 prediction error·RMSE | 불가 | 현재 endpoint-only protocol은 식별 불가 반례로 유지; raw history protocol을 별도 재정의 |
| NASA milling | wear estimation CNN/LSTM/TCN, cutting-condition-aware CNN | VB tool-wear, fixed machining-case split | 불가 | PP의 material-transfer RUL split과 분리하고, standard VB wear benchmark를 별도 수행 |

## 현재 최종 PP와의 내부 직접 비교

현재 final table에서는 Ridge, MLP/ResNet, boosting, FT/sequence Transformer, V-REx, GroupDRO, monotone NN, linear-tail RBF, Engression, GP, 그리고 가능한 설정의 TabPFN을 동일 strict test에서 비교한다. dual-scale PP로 MICH를 복구한 뒤 양의 R² 9개 설정에서는 현재 실행된 최고 비-PP 모델보다 최종 PP가 높다. 이는 **우리 프로토콜 안의 비교 결론**이고, 문헌 SOTA를 이겼다는 뜻은 아니다.

## 문헌 근거와 인용 후보

1. BatteryML은 HUST와 MATR을 포함한 다중 배터리 RUL benchmark 및 linear·statistical·deep baseline을 공개한다. 이 benchmark의 지표는 MAE이므로 현재 PP pooled R²와 직접 대조하지 않는다. https://github.com/microsoft/BatteryML
2. BatteryLife는 여러 배터리 dataset의 life-prediction benchmark와 공개 구현을 제공한다. https://github.com/Ruifeng-Tan/BatteryLife
3. Attia et al.은 MATR 계열에서 간결한 feature 기반 battery lifetime prediction을 제시했다. https://arxiv.org/abs/2101.01885
4. NASA B0005/6/7/18을 쓴 CNN-augmented sequential RUL 비교는 Transformer가 accelerated degradation에서 불안정할 수 있음을 보고한다. 이는 현재 PP의 strict tail setting과 같지는 않다. https://pmc.ncbi.nlm.nih.gov/articles/PMC12752999/
5. N-CMAPSS의 최근 CruiseBench는 LSTM·GRU·TCN·TSMixer를 비교하지만 별도의 cruise-stage mask와 RMSE/NASA score를 사용한다. https://arxiv.org/abs/2607.19380
6. XJTU-SY의 enhanced Transformer 계열은 raw two-axis vibration으로 RUL을 예측한다. https://pmc.ncbi.nlm.nih.gov/articles/PMC11481647/
7. NASA milling은 원래 flank wear VB를 목표로 하는 tool-wear benchmark다. https://catalog.data.gov/dataset/milling-wear

## PP 논문의 올바른 비교 문장

“최종 PP는 본 연구가 정의한 strict extrapolation protocol에서 강한 tabular·sequence·OOD 경쟁모델보다 높은 pooled R²를 보였다. 기존 문헌 SOTA는 관측 시점, 입력, unit split, 평가 지표가 달라 직접 수치 대조하지 않았으며, 대표 문헌 모델은 동일 protocol 재실행 대상으로 제시한다.”
